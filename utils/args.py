import argparse

def make_parser():
    parser = argparse.ArgumentParser("OC-SORT parameters")
    parser.add_argument("--expn", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed
    parser.add_argument( "--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--output_dir", type=str, default="evaldata/trackers/mot_challenge")
    parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("-d", "--devices", default=None, type=int, help="device for training")

    parser.add_argument("--local_rank", default=0, type=int, help="local rank for dist training")
    parser.add_argument( "--num_machines", default=1, type=int, help="num of node for training")
    parser.add_argument("--machine_rank", default=0, type=int, help="node rank for multi-node training")
    parser.add_argument("--model_type", type=str, required = True, help="Model type")
    
    
    ###
    parser.add_argument("--dataset_name", type=str, required = True, help="Dataset name")
    parser.add_argument("--model_path", type = str, required = True, help = "Path for the mamba tracker model")
    parser.add_argument("--association", type = str, required = True)
    parser.add_argument("--max_window", type = int, required = True)
    parser.add_argument("--num_blocks", type = int, required = True)
    parser.add_argument("--b1", type =float, required = True)
    parser.add_argument("--b2", type =float, required = True)
    
    parser.add_argument('--virtual_track_buffer', type = int, required = True, help = "Virtual detections creation threshold")
    
    #################################### BOT SORT #####################
    
    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="use Re-ID flag.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default="./fast_reid/configs/dancetrack/sbs_S50.yml", type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default="./pretrained/dance_sbs_S50_2.pth", type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5, help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25, help='threshold for rejecting low appearance similarity reid matches')
     # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.6, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.1, type=float, help="lowest detection threshold valid for tracks")
    parser.add_argument("--new_track_thresh", default=0.7, type=float, help="new track thresh")

    ### DiffMOT/ Deep-OC-SORT
    parser.add_argument("--aw_param", type=float, default=0.5)
    parser.add_argument("--w_assoc_emb", type=float, default=0.75, help="Combine weight for emb cost")
    # CMC
    parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow |orb | ecc")
    parser.add_argument("--ablation", dest="ablation", default=False, action="store_true", help="ablation ")

    parser.add_argument("--benchmark", dest="benchmark", type=str, default='dancetrack', help="benchmark to evaluate: MOT17 | MOT20")
 

    ### DIFT Paramaters #####
    # parser.add_argument('--up_ft_index', default=7,  type=int, help='which upsampling block to extract the ft map')
    parser.add_argument('--up_ft_index', default=2,  type=int, help='which upsampling block to extract the ft map')
    
    # parser.add_argument('--ensemble_size', default=,  type=int, help='ensemble size for getting an image ft map')
    parser.add_argument('--ensemble_size', default=2,   type=int, help='ensemble size for getting an image ft map')
    
    parser.add_argument('--temperature', default=0.1, type=float, help='temperature for softmax')        
    parser.add_argument("--n_last_frames", type=int, default=7, help="number of preceeding frames")
    parser.add_argument("--size_mask_neighborhood", default=12, type=int,  help="We restrict the set of source nodes considered to a spatial neighborhood of the query node")
    # parser.add_argument('--t', default=21, type=int, help='t for diffusion')    
    parser.add_argument('--t', default=301, type=int, help='t for diffusion')    
    
    parser.add_argument("--topk", type=int, default=5, help="accumulate label from top k neighbors")



    ####################################################################################################################
    parser.add_argument(
        "-f", "--exp_file",
        default=None,
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument(
        "--fp16", dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.",)
    parser.add_argument("--trt", dest="trt", default=False, action="store_true", help="Using TensorRT model for testing.",)
    parser.add_argument("--test", dest="test", default=False, action="store_true", help="Evaluating on test-dev set.",)
    parser.add_argument("--speed", dest="speed", default=False, action="store_true", help="speed test only.",)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER,)
    
    # det args
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument("--conf", default=0.1, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--seed", default=None, type=int, help="eval seed")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.6, help="detection confidence threshold")
    parser.add_argument("--iou_thresh", type=float, default=0.3, help="the iou threshold in Sort for matching")
    parser.add_argument("--min_hits", type=int, default=3, help="min hits to create track in SORT")
    parser.add_argument("--inertia", type=float, default=0.2, help="the weight of VDC term in cost matrix")
    parser.add_argument("--deltat", type=int, default=3, help="time step difference to estimate direction")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.9, help="matching threshold for tracking")
    parser.add_argument('--min-box-area', type=float, default=100, help='filter out tiny boxes')
    parser.add_argument("--gt-type", type=str, default="_val_half", help="suffix to find the gt annotation")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--public", action="store_true", help="use public detection")
    parser.add_argument('--asso', default="iou", help="similarity function: iou/giou/diou/ciou/ctdis")
    parser.add_argument("--use_byte", dest="use_byte", default=False, action="store_true", help="use byte in tracking.")
    
    # for kitti/bdd100k inference with public detections
    parser.add_argument('--raw_results_path', type=str, default="exps/permatrack_kitti_test/",
        help="path to the raw tracking results from other tracks")
    parser.add_argument('--out_path', type=str, help="path to save output results")
    parser.add_argument("--dataset", type=str, default="mot", help="kitti or bdd")
    parser.add_argument("--hp", action="store_true", help="use head padding to add the missing objects during \
            initializing the tracks (offline).")

    # for demo video
    parser.add_argument("--demo_type", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument( "--path", default="./videos/demo.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument(
        "--device",
        default="cuda:1",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    return parser