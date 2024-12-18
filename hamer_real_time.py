import time
import math
import asyncio
import requests
import threading
import cv2
import torch
import numpy as np

from collections import defaultdict
from OneEuroFilter import OneEuroFilter
from scipy.spatial.transform import Rotation as R

from vitpose_model import ViTPoseModel
from detectron2 import model_zoo
from hamer.models import  load_hamer, DEFAULT_CHECKPOINT
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy


class CameraProcessor:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.rh_init_xyz = []
        self.results = defaultdict()
        self.rescale_factor = 2.0
        self.light_blue = (0.65098039,  0.74117647,  0.85882353)

        self.latest_image = None  # 用于保存最新的图像帧

        self.running = False
        self.thread = None  # 处理视频流的线程
        self.stop_event = threading.Event()

        # 启动异步获取相机帧的任务
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.fetch_frame('http://10.1.100.66:5003/video'))  # 异步获取视频流

        # 启动线程来运行事件循环
        self.event_loop_thread = threading.Thread(target=self.run_event_loop)
        self.event_loop_thread.start()

        # Download and load checkpoints, Setup HaMeR model
        self.model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load detector
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        self.detector = DefaultPredictor_Lazy(detectron2_cfg)

        # keypoint detector
        self.cpm = ViTPoseModel(self.device)

        # OneEuroFilter
        self.lh_position_filters = [OneEuroFilter(freq=5, mincutoff=0.25) for _ in range(3)]
        self.lh_orientation_filters = [[OneEuroFilter(freq=5, mincutoff=0.25) for _ in range(3)] for _ in range(3)]

        self.rh_position_filters = [OneEuroFilter(freq=5, mincutoff=0.25) for _ in range(3)]
        self.rh_orientation_filters = [[OneEuroFilter(freq=5, mincutoff=0.25) for _ in range(3)] for _ in range(3)]

        # Setup the renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        
    def run_event_loop(self):
        """ 运行事件循环 """
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def fetch_frame(self, stream_url):
        """ 异步获取视频流 """
        bytes_data = b""
        stream = requests.get(stream_url, stream=True)
        for chunk in stream.iter_content(chunk_size=1024):
            bytes_data += chunk
            a = bytes_data.find(b'\xff\xd8')
            b = bytes_data.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes_data[a:b + 2]
                bytes_data = bytes_data[b + 2:]
                img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.latest_image = img  # 保存最后的图像帧

    def get_latest_image(self):
        """ 获取最新的图像帧（同步方法） """
        return self.latest_image
            
    def run(self):
        image = self.get_latest_image()
        if image is None:
            self.results = defaultdict()
            return
        
        # Detect humans in image
        det_out = self.detector(image)
        img = image.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        if len(det_instances)<1:
            self.results = defaultdict()
            return
        
        sorted_scores = det_instances.scores.sort()
        valid_idx = sorted_scores.indices[-1]
        valid_mask = torch.tensor([False]*len(det_instances))
        valid_mask[valid_idx] = True
        pred_bboxes = det_instances.pred_boxes.tensor[valid_mask].cpu().numpy()
        pred_scores = det_instances.scores[valid_mask].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        vitposes = vitposes_out[0]

        # Use hands based on hand keypoint detections
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # Rejecting not confident detections
        bboxes = []
        is_right = []
        keyp = left_hand_keyp
        valid = keyp[:,2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(0)
        keyp = right_hand_keyp
        valid = keyp[:,2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
            bboxes.append(bbox)
            is_right.append(1)

        if len(bboxes) == 0:
            self.results = defaultdict()
            return
        
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(self.model_cfg, image, boxes, right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            # 提取结果
            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            all_verts = []
            all_cam_t = []
            all_right = []

            results = defaultdict()
            for n in range(batch['img'].shape[0]):
                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                if is_right==0:
                    results["lh_bones_kps"] = out["pred_keypoints_3d"][n].tolist()
                    results["lh_pred_cam_t_full"] = cam_t.tolist()
                    results["lh_global_orient"] = out['pred_mano_params']["global_orient"][n][0].tolist()
                elif is_right==1:
                    results["rh_bones_kps"] = out["pred_keypoints_3d"][n].tolist()
                    results["rh_pred_cam_t_full"] = cam_t.tolist()
                    results["rh_global_orient"] = out['pred_mano_params']["global_orient"][n][0].tolist()

            # 保存当前帧的结果
            self.filtering_results(results)
            self.results = results

            # 特殊值打印及可视化
            self.debug_info(img, all_verts, all_cam_t, all_right, img_size, scaled_focal_length, vis_enable=False)

        return
    
    def filtering_results(self, results):
        if "lh_bones_kps" in results.keys():
            for i in range(3):
                results["lh_pred_cam_t_full"][i] = self.lh_position_filters[i](results["lh_pred_cam_t_full"][i])
                for j in range(3):
                    results["lh_global_orient"][i][j] = self.lh_orientation_filters[i][j](results["lh_global_orient"][i][j])

            temp = np.tile(results["lh_pred_cam_t_full"], (len(results["lh_bones_kps"]), 1)) + results["lh_bones_kps"]
            results["lh_bones_kps_cam"] = temp.tolist()
        else:
            self.lh_position_filters = [OneEuroFilter(freq=5, mincutoff=0.25) for _ in range(3)]
            self.lh_orientation_filters = [[OneEuroFilter(freq=5, mincutoff=0.25) for _ in range(3)] for _ in range(3)]
        
        if "rh_bones_kps" in results.keys():
            for i in range(3):
                results["rh_pred_cam_t_full"][i] = self.rh_position_filters[i](results["rh_pred_cam_t_full"][i])
                for j in range(3):
                    results["rh_global_orient"][i][j] = self.rh_orientation_filters[i][j](results["rh_global_orient"][i][j])

            temp = np.tile(results["rh_pred_cam_t_full"], (len(results["rh_bones_kps"]), 1)) + results["rh_bones_kps"]
            results["rh_bones_kps_cam"] = temp.tolist()
        else:
            self.rh_position_filters = [OneEuroFilter(freq=5, mincutoff=0.25) for _ in range(3)]
            self.rh_orientation_filters = [[OneEuroFilter(freq=5, mincutoff=0.25) for _ in range(3)] for _ in range(3)]

    def set_rh_init_xyz(self):
        rh_bones_kps_cam = self.results.get("rh_bones_kps_cam", [])

        if len(rh_bones_kps_cam)>0:
            self.rh_init_xyz = rh_bones_kps_cam[0]
            return True
        
        return False
    
    def debug_info(self, img, all_verts, all_cam_t, all_right, img_size, scaled_focal_length, vis_enable=False):
        lh_bones_kps_cam = self.results.get("lh_bones_kps_cam", [])
        rh_bones_kps_cam = self.results.get("rh_bones_kps_cam", [])
        rh_global_orient = self.results.get("rh_global_orient", [])
        if len(lh_bones_kps_cam)>0:
            print("left hand --- {}".format(lh_bones_kps_cam[0]))
        if len(rh_bones_kps_cam)>0:
            print("right hand --- {}".format(rh_bones_kps_cam[0]))
            print("right global orient --- {}".format(rh_global_orient))

        if vis_enable and len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=self.light_blue,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = self.renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[0], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            # info 1
            log_info_r_hand_0_gap = []
            if "rh_bones_kps_cam" in self.results.keys() and len(self.rh_init_xyz)>0:
                r_hand_0_gap = np.round([a - b for a, b in zip(self.results["rh_bones_kps_cam"][0], self.rh_init_xyz)], 3)
                log_info_r_hand_0_gap = [-1*r_hand_0_gap[2], r_hand_0_gap[0], -1*r_hand_0_gap[1]]

            if len(self.rh_init_xyz)<1:
                _ = self.set_rh_init_xyz()
            
            # info 2
            rh_thumb_finger = self.results["rh_bones_kps"][4]
            rh_index_finger = self.results["rh_bones_kps"][8]
            rh_distance = math.sqrt((rh_index_finger[0] - rh_thumb_finger[0])**2 + (rh_index_finger[1] - rh_thumb_finger[1])**2 + (rh_index_finger[2] - rh_thumb_finger[2])**2)
            rh_distance = round(rh_distance, 3)

            cv2.putText(input_img_overlay, str(log_info_r_hand_0_gap), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 3, cv2.INTER_AREA)
            cv2.imshow('Camera', input_img_overlay)
            cv2.waitKey(1)

    def process_stream(self):
        while self.running:
            # 检查停止事件
            if self.stop_event.is_set():
                break
            self.run()

    def start(self):
        """ 启动处理器，持续返回run方法的变量 """
        self.running = True
        self.stop_event.clear()  # 确保停止事件被清除
        
        self.thread = threading.Thread(target=self.process_stream, args=())
        self.thread.start()

    def stop(self):
        """ 停止处理器 """
        self.running = False
        self.stop_event.set()  # 设置停止事件
        # 等待线程完成
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()  # 等待线程执行完毕
        self.thread = None

    def get_results(self):
        """ 获取最新的处理结果 """
        return self.results


if __name__ == '__main__':
    processor = CameraProcessor()
    time.sleep(1)
    processor.start()
    time.sleep(5)
    _ = processor.set_rh_init_xyz()
    # time.sleep(1)
    # result = processor.get_results()
    # print(result.get('rh_bones_kps_cam', []))
    # time.sleep(1)
    # result = processor.get_results()
    # print(result.get('rh_bones_kps_cam', []))
    # time.sleep(3)
    # processor.stop()
