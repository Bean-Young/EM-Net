def Get_field(path):
    import numpy as np
    import torch
    from types import SimpleNamespace
    import sys
    from MM.utils import patchstack
    from skimage import transform
    sys.path.append('..')
    from MM.MM_Module import MM_Modules
    from MM.pretreatment import pretreatment_for_fields
    opts = SimpleNamespace()

    opts.R                        = 64
    opts.stride                   = 64
    opts.eta                      = 0.01
    opts.delta                    = 0.05
    opts.lr_angles                = 0.003
    opts.lr_x0y0                  = 0.03
    opts.lambda_boundary_final    = 0.5
    opts.lambda_color_final       = 0.1
    opts.nvals                    = 10
    opts.num_initialization_iters = 30
    opts.num_refinement_iters     = 1000
    opts.greedy_step_every_iters  = 40
    opts.parallel_mode            = True

    i = 0
    lis = [0]
    while i < 1056:
        i = i + 65
        lis.append(i)
        i = i + 1
        lis.append(i)

    def foj_optimize_verbose():
        for i in range(foj.num_iters):
            foj.step(i)


    clean_img=pretreatment_for_fields(path)
    clean_img=transform.resize(clean_img, (1024, 1024),order=0)
    img = clean_img + np.random.randn(*clean_img.shape)
    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn)

    foj = MM_Modules(img, opts)
    foj_optimize_verbose()


    params = torch.cat([foj.angles, foj.x0y0], dim=1)
    dists, _, patches = foj.get_dists_and_patches(params)
    local_boundaries = foj.dists2boundaries(dists)
    boundary_vis = patchstack(local_boundaries)[0, :, :, :].detach().permute(1, 2, 0).cpu().numpy()
    boundary_vis=np.squeeze(boundary_vis,axis = None)

    rows_to_keep = [i for i in range(boundary_vis.shape[0]) if i not in lis]
    cols_to_keep = [i for i in range(boundary_vis.shape[1]) if i not in lis]

    result = boundary_vis[np.ix_(rows_to_keep, cols_to_keep)]


    return result

