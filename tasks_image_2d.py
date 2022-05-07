from utils.image_utils import init_source_target
from custom_types import *
from models import encoding_controler, encoding_models
from utils import files_utils, train_utils, image_utils
import constants
from torch.utils.data import TensorDataset


def plot_image(model: encoding_controler.EncodedController, vs_in: T, ref_image: ARRAY):
    model.eval()
    with torch.no_grad():
        if model.is_progressive:
            out, mask = model(vs_in, get_mask=True)
            if mask.dim() != out.dim():
                mask: T = mask.unsqueeze(0).expand(out.shape[0], mask.shape[0])
            hm = mask.sum(1) / mask.shape[1]
            hm = image_utils.to_heatmap(hm)
            hm = hm.view(*ref_image.shape[:-1], 3)
        else:
            out = model(vs_in, get_mask=True)
            hm = None
        out = out.view(ref_image.shape)
    model.train()
    return out, hm


def optimize(image_path: Union[ARRAY, str], encoding_type: EncodingType, model_params,
             controller_type: ControllerType, control_params: encoding_controler.ControlParams, group, device: D,
             freq: int, verbose=False):

    def shuffle_coords():
        nonlocal vs_in, labels
        order = torch.rand(vs_in.shape[0]).argsort()
        vs_in, labels = vs_in[order], labels[order]

    patch_size = 65536


    name = files_utils.split_path(image_path)[1]
    vs_base, vs_in, labels, target_image, image_labels, masked_image = group
    tag = f'{name}_{encoding_type.value}_{controller_type.value}'
    out_path = f'{constants.CHECKPOINTS_ROOT}/2d_images/{name}/'
    lr = 1e-3
    model = encoding_controler.get_controlled_model(model_params, encoding_type, control_params, controller_type).to(device)
    block_iterations = model.block_iterations
    train_ds = TensorDataset(vs_in, labels)
    train_dl = DataLoader(train_ds, batch_size=patch_size, shuffle=True, num_workers=4)
    # vs_in, labels = vs_in.to(device), labels.to(device)
    # vs_base, image_labels = vs_base.to(device), image_labels.to(device)
    opt = Optimizer(model.parameters(), lr=lr)
    # check number of trainable parameters in the model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The number of parameters in this model is {pytorch_total_params}.")
    print(f"The number of trainable parameters in this model is {pytorch_total_trainable_params}.")
    #
    logger = train_utils.Logger().start(control_params.num_iterations, tag=tag)
    files_utils.export_image(target_image, f'{out_path}target.png')
    if masked_image is not None:
        files_utils.export_image(masked_image, f'{out_path}target_masked.png')
    for j in range(control_params.num_iterations):
        for xb, yb in train_dl:
            print(xb.shape)
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss_all = nnf.mse_loss(out, yb, reduction='none')
            loss = loss_all.mean()
            loss.backward()
            opt.step()
            model.stash_iteration(loss_all.mean(-1))
            logger.stash_iter('mse_train', loss)
        shuffle_coords()
        if block_iterations > 0 and (j + 1) % block_iterations == 0:
            model.update_progress()
        if (j + 1) == control_params.num_iterations and verbose:
            with torch.no_grad():
                out, hm = plot_image(model, vs_base, target_image)
                if hm is not None:
                    files_utils.export_image(hm, f'{out_path}heatmap_{tag}/{j:04d}.png')
                files_utils.export_image(out, f'{out_path}opt_{tag}/{j:04d}.png')
                # model = model.to(device)
        logger.reset_iter()
    logger.stop()
    files_utils.save_model(model, f'{out_path}model_{tag}.pth')
    # if verbose:
    #     # image_utils.gifed(f'{out_path}opt_{tag}/', .07, tag, reverse=False)
    #     if model.is_progressive:
    #         image_utils.gifed(f'{out_path}heatmap_{tag}/', .07, tag, reverse=False)
    #         files_utils.delete_all(f'{out_path}heatmap_{tag}/', '.png',
    #                                filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])
    #     files_utils.delete_all(f'{out_path}opt_{tag}/', '.png',
    #                            filter_out=lambda x: f'{control_params.num_iterations - 1}' == x[1])


def main() -> int:
    device = CUDA(0)
    image_path = files_utils.get_source_path()
    name = files_utils.split_path(image_path)[1]
    scale = .25
    group = init_source_target(image_path, name, scale=scale, max_res=4333, square=False)
    model_params = encoding_models.ModelParams(domain_dim=2, output_channels=3, num_freqs=256,
                                               hidden_dim=256, std=20., num_layers=3)
    control_params = encoding_controler.ControlParams(num_iterations=5000, epsilon=1e-3, res=128)
    # encoding_types = (EncodingType.NoEnc, EncodingType.FF, EncodingType.FF)
    # controller_types = (ControllerType.NoControl, ControllerType.NoControl, ControllerType.SpatialProgressionStashed)
    encoding_type = EncodingType.FF
    controller_type = ControllerType.SpatialProgressionStashed
    # for encoding_type, controller_type in zip(encoding_types, controller_types):
    #     optimize(image_path, encoding_type, model_params, controller_type, control_params, group, device,
    #              50, verbose=True)
    optimize(image_path, encoding_type, model_params, controller_type, control_params, group, device,
                50, verbose=True)
    return 0


if __name__ == '__main__':
    exit(main())
