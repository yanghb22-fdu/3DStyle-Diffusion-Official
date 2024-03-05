import os
from tqdm import tqdm
import torch
from neural_style_field import NeuralStyleField
from utils import device
# from utils import clip_model
import numpy as np
import random
import torchvision
import os
import argparse
from pathlib import Path
from torchvision import transforms
import open3d as o3d

import cv2

def get_normalize_mesh(pro_path):
    mesh = o3d.io.read_triangle_mesh(pro_path)
    vertices = np.asarray(mesh.vertices)
    shift = np.mean(vertices,axis=0)
    scale = np.max(np.linalg.norm(vertices-shift, ord=2, axis=1))
    vertices = (vertices-shift) / scale
    mesh.vertices = o3d.cuda.pybind.utility.Vector3dVector(vertices)
    return mesh
    
def train(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float32)
    # torch.set_num_threads(8)
    # Constrain all sources of randomness
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    objbase, extension = os.path.splitext(os.path.basename(args.obj_path))
    # Check that isn't already done
    if (not args.overwrite) and os.path.exists(os.path.join(args.output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj")):
        print(f"Already done with {args.output_dir}")
        exit()
    elif args.overwrite and os.path.exists(os.path.join(args.output_dir, "loss.png")) and \
            os.path.exists(os.path.join(args.output_dir, f"{objbase}_final.obj")):
        import shutil
        for filename in os.listdir(args.output_dir):
            file_path = os.path.join(args.output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    # Save config
    with open(os.path.join(args.output_dir, "args.txt"), "w") as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    losses = []
    n_augs = args.n_augs
    dir = args.output_dir
    # global transformation
    augment_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(1, 1)), #Obtain a thumbnail image to meet the requirements of clip's input image size
    ])
    # local transformation
    normaugment_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(args.mincrop, args.maxcrop)),
    ])

    if args.n_normaugs > 0:
        import clip
        from utils import clip_model
    
    # get diffusion model or controlnet stable diffusion
    if args.guidance == 'stable-diffusion':
        from sd import StableDiffusion
        guidance = StableDiffusion(device, args.sd_version, args.hf_key)
        text_z = guidance.get_text_embeds([args.prompt], [''])
        
    elif args.guidance == 'control-stable-diffusion':
        from control_sd import ControlStableDiffusion
        guidance = ControlStableDiffusion(device, args.sd_version, args.control_version, args.hf_key)
        text_z = guidance.get_text_embeds([args.prompt + ',' + args.a_prompt], [args.n_prompt])
        

    normweight = 1.0
    model = NeuralStyleField(args.material_random_pe_numfreq,
                             args.material_random_pe_sigma,
                             args.num_lgt_sgs,
                             args.max_delta_theta,
                             args.max_delta_phi,
                             args.normal_nerf_pe_numfreq,
                             args.normal_random_pe_numfreq,
                             args.symmetry,
                             args.radius,
                             args.background,
                             args.init_r_and_s,
                             args.width,
                             args.init_roughness,
                             args.init_specular,
                             args.material_nerf_pe_numfreq,
                             args.normal_random_pe_sigma,
                             args.if_normal_clamp
                            )
    if torch.cuda.is_available():
        model.cuda()

     
    model.train()
    optim = torch.optim.AdamW(model.parameters(), args.learning_rate)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                        [500,1000],
                                                        args.lr_decay)
   
    if args.prompt:
        prompt = args.prompt
        # Save prompt
        with open(os.path.join(dir, prompt), "w") as f:
            f.write("")

        if args.n_normaugs > 0:
            prompt_token = clip.tokenize([prompt]).to(device)
            encoded_text = clip_model.encode_text(prompt_token)
            norm_encoded = encoded_text
    # ipdb.set_trace()

    mesh = get_normalize_mesh(args.obj_path)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh)
    trainer = tqdm(range(args.n_iter))
    
    # Get training azim and elev
    save_pose_path = os.path.join(dir, "save_poses")
    if not os.path.exists(save_pose_path):
        os.makedirs(save_pose_path)

    train_azim = []
    train_elev = []
    
    for i in trainer:
        optim.zero_grad()

        
        
        rendered_images, depths, azim, elev = model(scene=scene, 
                                num_views=args.n_views,
                                center_azim=args.frontview_center[0],
                                center_elev=args.frontview_center[1],
                                std=args.frontview_std,
                                train_poses = args.train_poses,
                                train_render_elev = args.train_render_elev,
                                )
        rendered_images = rendered_images.cuda()                                                # (1, 4, 512, 512)

        if i % 100 == 0:

            # Save tarin azim elev
            train_azim.append(azim)
            train_elev.append(elev)
            save_azim = torch.Tensor(train_azim)
            save_elev = torch.Tensor(train_elev)
            torch.save(save_azim, os.path.join(save_pose_path, "azim_{}.pt".format(i)))
            torch.save(save_elev, os.path.join(save_pose_path, "elev_{}.pt".format(i)))


            # report_process(args, dir, i, loss, loss_check, losses, rendered_images[:,0:3,:,:])
            print('iter: {} loss: {}'.format(i, np.mean(losses[-100:])))
            torchvision.utils.save_image(rendered_images[:,0:3,:,:], os.path.join(dir, 'iter_{}.jpg'.format(i)))
            if args.guidance == 'control-stable-diffusion':
                # torchvision.utils.save_image(torch.from_numpy(depths).permute(0, 3, 1, 2)[:,0:3,:,:], os.path.join(dir, 'iter_{}_depth.jpg'.format(i)))
                if args.n_views == 1:
                    depth = depths[0]
                    cv2.imwrite(os.path.join(dir, 'iter_{}_depth.jpg'.format(i)), depth)
                else:
                    for j in range(args.n_views):
                        depth = depths[j]
                        cv2.imwrite(os.path.join(dir, 'iter_{}_depth_{}.jpg'.format(i, j)), depth)

            if i % 1000 == 0:
                torch.save({'model': model.state_dict()}, os.path.join(dir, f'iter{i:03d}.pth'))
            
            # evaluation
            if i == args.n_iter-1:
                
                model.eval()
                eval_path = os.path.join(dir, "eval_results")
                if not os.path.exists(eval_path):
                    os.makedirs(eval_path)
                dir_rgb = os.path.join(eval_path, 'rgb')
                dir_normal1 = os.path.join(eval_path, 'normal1')
                dir_normal2 = os.path.join(eval_path, 'normal2')
                dir_roughness = os.path.join(eval_path, 'roughness')
                dir_diffuse = os.path.join(eval_path, 'diffuse')
                dir_specular = os.path.join(eval_path, 'specular')
                if not os.path.exists(dir_rgb):
                    os.makedirs(dir_rgb)
                if not os.path.exists(dir_normal1):
                    os.makedirs(dir_normal1)
                if not os.path.exists(dir_normal2):
                    os.makedirs(dir_normal2)
                if not os.path.exists(dir_roughness):
                    os.makedirs(dir_roughness)
                if not os.path.exists(dir_diffuse):
                    os.makedirs(dir_diffuse)
                if not os.path.exists(dir_specular):
                    os.makedirs(dir_specular)


                view_num = 100
                from eval import save_gif
                azim = torch.linspace(0, 2 * np.pi + 0, view_num)  # since 0 = 2π dont include last element
                elev = torch.tensor(args.frontview_center[1])    
                for j in tqdm(range(view_num)):   
                
                    rendered_images ,normal1 , normal2 ,roughness, diffuse, specular= model.render_single_image(scene=scene, 
                                                azim=azim[j],
                                                elev=elev,
                                                r=args.radius
                                                )
                    torchvision.utils.save_image(rendered_images, os.path.join(dir_rgb, f'iter_test_rgb_{j:03d}.jpg'))
                    torchvision.utils.save_image(normal1, os.path.join(dir_normal1, f'iter_test_normal1_{j:03d}.jpg'))
                    torchvision.utils.save_image(normal2, os.path.join(dir_normal2, f'iter_test_normal2_{j:03d}.jpg'))
                    torchvision.utils.save_image(roughness, os.path.join(dir_roughness, f'iter_test_roughness_{j:03d}.jpg'))
                    torchvision.utils.save_image(diffuse, os.path.join(dir_diffuse, f'iter_test_diffuse_{j:03d}.jpg'))
                    torchvision.utils.save_image(specular, os.path.join(dir_specular, f'iter_test_specular_{j:03d}.jpg'))
                    del rendered_images ,normal1 , normal2 ,roughness, diffuse, specular
                save_gif(dir_rgb,30)
                save_gif(dir_normal1,30)
                save_gif(dir_normal2,30)
                save_gif(dir_roughness,30)
                save_gif(dir_specular,30)
                save_gif(dir_diffuse,30)
        
        if i < args.n_iter-1:

            if args.n_views == 1:
                loss = 0.0
                
                if args.guidance == 'stable-diffusion':
                    augmented_images = augment_transform(rendered_images[:,0:3,:,:])
                    select_view = np.random.choice(range(args.n_views))
                    augmented_image = augmented_images[select_view].unsqueeze(0)
                    loss = guidance.train_step(text_z, augmented_image)
                    
                elif args.guidance == 'control-stable-diffusion':
                    # get depth_image
                    augmented_images = rendered_images[:,0:3,:,:]                                   # (1, 3, 512, 512)
                    select_view = np.random.choice(range(args.n_views))
                    augmented_image = augmented_images[select_view].unsqueeze(0)                    # (1, 3, 512, 512)
                    depth = depths[select_view]                                                     # (512, 512, 3) 0-255
                    depth = torch.tensor(depth)
                    depth = depth.unsqueeze(0)                                                      # (1, 512, 512, 3)
                    depth = depth / 255.0
                    depth = depth.permute(0, 3, 1, 2)                                               # (1, 3, 512, 512)                
                    
                    loss = guidance.train_step(text_z, augmented_image, depth)
                
                # loss.backward(retain_graph=True)
                loss.backward()
            
            else:
                loss = 0.0
                for j in range(args.n_views):
                    augmented_images = rendered_images[:,0:3,:,:]                                   # (n, 3, 512, 512)
                    augmented_image = augmented_images[j].unsqueeze(0)                              # (1, 3, 512, 512)
                    depth = depths[j]
                    depth = torch.tensor(depth)
                    depth = depth.unsqueeze(0)
                    depth = depth / 255.0
                    depth = depth.permute(0, 3, 1, 2)

                    loss += guidance.train_step(text_z, augmented_image, depth)
                # loss = loss / args.n_views
                loss.backward(retain_graph=True)

            if args.n_normaugs > 0:
                normloss = 0.0
                for _ in range(args.n_normaugs):
                    augmented_image = normaugment_transform(rendered_images)                                        # (2, 4, 224, 224)
                    shape = augmented_image.shape[0]*augmented_image.shape[2]*augmented_image.shape[3]              # 2 * 224 * 224
                    object_percent = torch.sum(augmented_image[:,3,:,:]==1) / shape
                    while object_percent <= args.local_percentage: 
                        augmented_image = normaugment_transform(rendered_images)
                        object_percent = torch.sum(augmented_image[:,3,:,:]==1) / shape

                    augmented_image = augmented_image[:,0:3,:,:]                                                   # (1, 3, 224, 224)
                    if i % 50 == 0:
                        torchvision.utils.save_image(augmented_image, os.path.join(dir, 'iter_local{}.jpg'.format(i)))
                    encoded_renders = clip_model.encode_image(augmented_image)
                    if args.prompt:
                        if args.clipavg == "view":
                            if norm_encoded.shape[0] > 1:
                                normloss -= normweight * torch.cosine_similarity(torch.mean(encoded_renders, dim=0),
                                                                                    torch.mean(norm_encoded, dim=0),
                                                                                    dim=0)
                            else:
                                normloss -= normweight * torch.cosine_similarity(
                                    torch.mean(encoded_renders, dim=0, keepdim=True),
                                    norm_encoded)
                        else:
                            normloss -= normweight * torch.mean(
                                torch.cosine_similarity(encoded_renders, norm_encoded))
                normloss.backward(retain_graph=True)

            optim.step()
            lr_scheduler.step()
            with torch.no_grad():
                losses.append(loss.item())
            if args.decayfreq is not None:
                if i % args.decayfreq == 0:
                    normweight *= args.cropdecay
            lr = optim.state_dict()['param_groups'][0]['lr']
            trainer.set_description(desc=f'lr:{lr}')
            del rendered_images, augmented_images, augmented_image
            torch.cuda.empty_cache()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_lgt_sgs', type=int, default=32) #the number of light SGs
    parser.add_argument('--max_delta_theta', type=float, default=1.5707) #maximum offset of elevation angle whose unit is radian
    parser.add_argument('--max_delta_phi', type=float, default=1.5707) #maximum offset of azimuth angle whose unit is radian
    
    parser.add_argument('--normal_nerf_pe_numfreq',  type=int, default=0) #the number of frequencies using nerf's position encoding in normal network
    parser.add_argument('--normal_random_pe_numfreq', type=int, default=0) #the number of frequencies using random position encoding in normal network
    parser.add_argument('--normal_random_pe_sigma', type=float, default=20.0) #the sigma of random position encoding in normal network
    parser.add_argument('--material_nerf_pe_numfreq',  type=int, default=0) #the numer of frequencies using nerf's position encoding in svbrdf network
    parser.add_argument('--material_random_pe_numfreq', type=int, default=0) #the numer of frequencies using random position encoding in svbrdf network
    parser.add_argument('--material_random_pe_sigma', type=float, default=20.0) #the sigma of random position encoding in svbrdf network
    parser.add_argument('--if_normal_clamp', action='store_true') 
     
    parser.add_argument('--init_r_and_s', action='store_true') #It will initialize roughness and specular if setting true
    parser.add_argument('--init_roughness', type=float, default=0.7) #Initial value of roughness 0~1
    parser.add_argument('--init_specular', type=float, default=0.23)  #Initial value of specular 0~1
    parser.add_argument('--width', type=int, default=512) #the size of render image will be [width,width]
    
    parser.add_argument('--radius', type=float, default=2.0) #the sampling raidus of camara position
    parser.add_argument('--background', type=str, default='black') #the background of render image.'black','white' or 'gaussian' can be selected
    parser.add_argument('--local_percentage',type=float, default=0.7) #percent threshold of the object's mask in cropped image.It will be cropped again
                                                                      #if the proportion of the object's mask in cropped image is less than this threshold.
                                                                      #This parameter can effectively prevent image degradation
    parser.add_argument('--obj_path', type=str, default='meshes/mesh1.obj') #the storage path of raw or original mesh
    parser.add_argument('--prompt', type=str, default='a pig with pants') #the text prompt to style a raw mesh
    parser.add_argument('--output_dir', type=str, default='round2/alpha5') #directory where the results will be saved
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--lr_decay', type=float, default=1) #decay factor of learning rate
    parser.add_argument('--n_views', type=int, default=4) #number of viewpoints optimized at the same time in an iteration
    parser.add_argument('--n_augs', type=int, default=0) #In one iteration, the gradient retrieval times of the rendered thumbnail
    parser.add_argument('--n_normaugs', type=int, default=0) #In one iteration, the gradient retrieval times of the local clip of the rendered image
    parser.add_argument('--n_iter', type=int, default=1501) #number of iterations

    parser.add_argument('--frontview_std', type=float, default=8) # Angular variance of the off-center view
    parser.add_argument('--frontview_center', nargs=2, type=float, default=[0., 0.]) #Center position of viewpoint.[azimuth angle(0~2π),elevation angle(0~π)]
    parser.add_argument('--clipavg', type=str, default=None)
    parser.add_argument('--mincrop', type=float, default=1) #minimium clipping scale in 2D augmentation 
    parser.add_argument('--maxcrop', type=float, default=1) #maximium clipping scale in 2D augmentation
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--seed', type=int, default=0) #random seed
    parser.add_argument('--symmetry', default=False, action='store_true') #With this symmetry prior, the texture of the mesh 
                                                                          #will be symmetrical along the z-axis.We use this parameter in person
    parser.add_argument('--decayfreq', type=int, default=None) #decay freaquency of learning rate
    
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--sd_version', type=str, default='1.5', help="stable diffusion version")
    parser.add_argument('--control_version', type=str, default="fusing/stable-diffusion-v1-5-controlnet-depth")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--guidance', type=str, default='control-stable-diffusion', help='choose from [control-stable-diffusion, stable-diffusion, clip]')
    
    parser.add_argument("--a_prompt", type=str, default="best quality, extremely detailed")
    parser.add_argument("--n_prompt", type=str, default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality")

    parser.add_argument("--train_poses", type=str, default='default')   # default==tango; uniform, random 
    parser.add_argument('--train_render_elev', nargs=3, type=float, default=[5.6549, 0.0, 0.6283]) #elevation angle(0~π)] during training
    
    args = parser.parse_args()
 
    train(args)

   
