import argparse
import copy
import os
import time
import warnings

import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist
from mmdet.apis import set_random_seed, train_detector
from mmdet.utils import collect_env, get_device, get_root_logger, setup_multi_processes

from nets.nn import build_detector
from utils.dataset import build_dataset

warnings.filterwarnings("ignore")


def train(args):
    cfg = Config.fromfile(args.config)
    # set multiprocess settings
    setup_multi_processes(cfg)

    exp_name = os.path.splitext(os.path.basename(args.config))[0]

    cfg.work_dir = os.path.join('./weights', exp_name)
    cfg.gpu_ids = [0]
    # init distributed env first, since logger depends on the dist info.
    if args.distributed:
        init_dist('pytorch', **cfg.dist_params)
        cfg.gpu_ids = range(args.world_size)
    # create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    # dump config
    cfg.dump(os.path.join(cfg.work_dir, os.path.basename(args.config)))
    # init the logger before other steps
    log_file = os.path.join(cfg.work_dir, f'{exp_name}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info = '\n'.join([f'{k}: {v}' for k, v in collect_env().items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {args.distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    cfg.device = get_device()

    cfg.seed = args.local_rank
    meta['seed'] = args.local_rank
    meta['exp_name'] = os.path.basename(args.config)

    model = build_detector(cfg.model, cfg.get('train_cfg'), cfg.get('test_cfg'))

    model.init_weights()

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(mmdet_version=collect_env()['MMDetection'],
                                          CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_detector(model, datasets, cfg, args.distributed, True, exp_name, meta)


def test(args):
    from mmdet.apis import multi_gpu_test, single_gpu_test
    from mmdet.datasets import build_dataloader, replace_ImageToTensor
    from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
    from mmcv.runner import get_dist_info, load_checkpoint, wrap_fp16_model

    cfg = Config.fromfile(args.config)
    # set multiprocess settings
    setup_multi_processes(cfg)
    # in case the test dataset is concatenated
    samples_per_gpu = cfg.data.samples_per_gpu
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    if args.distributed:
        # init distributed env first, since logger depends on the dist info.
        init_dist('pytorch', **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=samples_per_gpu,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=args.distributed, shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    wrap_fp16_model(model)
    config_name = os.path.basename(args.config)
    config_name = os.path.splitext(config_name)[0]
    checkpoint = load_checkpoint(model, f'./weights/{config_name}/{config_name}.pth', 'cpu')

    model.CLASSES = checkpoint['meta']['CLASSES']
    if args.distributed:
        model = MMDistributedDataParallel(model.cuda(),
                                          device_ids=[torch.cuda.current_device()],
                                          broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, gpu_collect=True)
        if get_dist_info()[0] == 0:
            dataset.format_results(outputs, jsonfile_prefix=f"./weights/{config_name}")
    else:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader)
        dataset.format_results(outputs, jsonfile_prefix=f"./weights/{config_name}")


def fps(args):
    from mmcv.runner import wrap_fp16_model
    from mmcv.parallel import MMDataParallel
    from mmcv.parallel import MMDistributedDataParallel
    from mmdet.datasets import build_dataloader, replace_ImageToTensor

    cfg = Config.fromfile(args.config)
    # set multiprocess settings
    setup_multi_processes(cfg)
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    if args.distributed:
        # init distributed env first, since logger depends on the dist info.
        init_dist('pytorch', **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=cfg.data.samples_per_gpu,
                                   workers_per_gpu=cfg.data.workers_per_gpu,
                                   dist=args.distributed, shuffle=False)

    # build the model
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.forward = model.forward_dummy
    wrap_fp16_model(model)

    inf_time = 0
    if args.distributed:
        model = MMDistributedDataParallel(model.cuda(),
                                          device_ids=[torch.cuda.current_device()],
                                          broadcast_buffers=False)
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    if args.local_rank == 0:
        print('calculating FPS ...')
    for i, data in enumerate(data_loader):
        start_time = time.perf_counter()
        with torch.no_grad():
            model(data['img'])
        if i > 10:
            inf_time += time.perf_counter() - start_time
    torch.cuda.synchronize()
    if args.local_rank == 0:
        print('FPS: ', int((len(dataset) - 10) / inf_time))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--fps', action='store_true')
    parser.add_argument('--local_rank', default=0, type=int)

    args = parser.parse_args()

    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    set_random_seed(0, True)

    if args.train:
        train(args)
    if args.test:
        test(args)
    if args.fps:
        fps(args)


if __name__ == '__main__':
    main()
