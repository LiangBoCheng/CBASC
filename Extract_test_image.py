import torch
import torch.utils.data
import utils
import numpy as np
import transforms as T
from bert.modeling_bert import BertModel
from lib import segmentation
from torchvision.transforms import functional as F
import cv2
import os

# Extract test images from rrsisd/RefSegRS/RISBench dataset

torch.cuda.set_device(0)

def get_dataset(image_set, transform, args):
    if args.dataset == "rrsisd":
        from data.rrsisd_refer_bert import ReferDataset
    elif args.dataset == "RISBench":
        from data.risbench_refer_bert import ReferDataset
    else:
        from data.refsegrs_refer_bert import ReferDataset
    ds = ReferDataset(args,
                      split=image_set,
                      image_transforms=transform,
                      target_transforms=None
                      )
    num_classes = 2

    return ds, num_classes


def evaluate(model, data_loader, bert_model, device):
    """仅输出测试图片"""
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    save_dir = "./test_images_only/"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for data in metric_logger.log_every(data_loader, 100, header="Test:"):
            image, target, sentences, attentions, img_name = data
            image = image.to(device)

            # 仅保存原图
            img_base = os.path.splitext(os.path.basename(img_name[0]))[0]
            save_path = os.path.join(save_dir, img_base)
            save_test_image(image, save_path)

            # 清理显存
            del image, target, sentences, attentions, img_name


def get_transform(args):
    transforms = [T.Resize(args.img_size, args.img_size),
                  T.ToTensor(),
                  T.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])]
    return T.Compose(transforms)


def save_test_image(image, save_path):
    """仅保存原图"""
    # 反归一化原图
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1 / s for s in std]
    im = F.normalize(image, mean=inv_mean, std=inv_std)
    im = im[0, :, :, :].cpu().detach().numpy()
    im = im.transpose([1, 2, 0])
    im = np.uint8(np.clip(im * 255, 0, 255))

    # 保存原图
    cv2.imwrite(save_path + ".png", cv2.cvtColor(im, cv2.COLOR_RGB2BGR))


def main(args):
    device = torch.device(args.device)
    dataset_test, _ = get_dataset(args.split, get_transform(args=args), args)

    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1,
                                                   sampler=test_sampler, num_workers=args.workers)
    print(args.model)

    # 模型加载部分仍保留
    single_model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    checkpoint = torch.load(args.resume, map_location=torch.device('cuda:0'))
    single_model.load_state_dict(checkpoint['model'], strict=False)
    model = single_model.to(device)

    if args.model != 'lavt_one':
        model_class = BertModel
        single_bert_model = model_class.from_pretrained(args.ck_bert)
        if args.ddp_trained_weights:
            single_bert_model.pooler = None
        single_bert_model.load_state_dict(checkpoint['bert_model'])
        bert_model = single_bert_model.to(device)
    else:
        bert_model = None

    # 调用评估，仅输出测试图片
    evaluate(model, data_loader_test, bert_model, device=device)


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    print('Image size: {}'.format(str(args.img_size)))
    main(args)