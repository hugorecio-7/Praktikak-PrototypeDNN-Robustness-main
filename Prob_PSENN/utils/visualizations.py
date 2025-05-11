import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import dataset_handler

# VISUALIZATION

def plt_size(width, height):
    plt.gcf().set_size_inches(width, height)

def plot_circle(rad):
    x = np.linspace(-rad - 1.0, rad + 1.0, 100)
    y = np.linspace(-rad - 1.0, rad + 1.0, 100)
    X, Y = np.meshgrid(x, y)
    F = X ** 2 + Y ** 2 - rad
    plt.contour(X, Y, F, [0])


def get_color_vec(n_classes=None):
    """Returns a vector of colors (10 colors by default). If a number of classes is specified, 
       the returned vector will be of that size.

    Args:
        n_classes (int, optional): Number of classes (and thus number of colors needed). Defaults to None.

    Returns:
        list: List containing the color specifiers (strings).
    """
    color_vec = ["green", "blue", "red", "orange", "gray", "purple", "pink", "black", "olive", "cyan"]
    if n_classes==None:
        return color_vec
    
    #Make sure that the number of colors matches the number of classes, if needed
    if len(color_vec) > n_classes:
        color_vec = color_vec[0:n_classes]
    elif len(color_vec) < n_classes:
        while len(color_vec) < n_classes:
            color_vec += color_vec
        color_vec = color_vec[0:n_classes]

    return color_vec


def visualize_gauss2d(gan, std_scaler, prior_sampler, examples, labels, save_name="/tmp/g2d.png", show=False):
    color_vec = get_color_vec()
    plt.figure(figsize=(18, 5))

    # Sample points from the "distribution manifold" in order to visualize the location of the classes
    nx = ny = 15
    x_values = np.linspace(.01, .99, nx)
    y_values = np.linspace(.01, .99, ny)
    plt.subplot(1, 3, 1)
    canvas = np.empty((28 * ny, 28 * nx))
    for i, xi in enumerate(x_values):
        for j, yi in enumerate(y_values):
            z_mu = np.array([[stats.norm.ppf(xi,0.0,std_scaler), stats.norm.ppf(yi,0.0,std_scaler)]]).astype('float32')
            x_mean = gan.decoder(z_mu)
            #canvas[(nx-i-1) * 28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].numpy().reshape(28,28)
            canvas[(ny-j-1)*28:(ny-j)*28, i*28:(i+1)*28] = x_mean[0].numpy().reshape(28, 28)
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.axis("off")
    plt.tight_layout()

    # Sample and plot the target distribution
    sample_size = 10000
    dist_z = prior_sampler(sample_size)
    plt.subplot(1, 3, 2)
    plt.scatter(dist_z[:, 0], dist_z[:, 1], s=2)
    for i, xi in enumerate(x_values):
        for j, yi in enumerate(y_values):
            z_mu = np.array([[stats.norm.ppf(xi,0.0,std_scaler), stats.norm.ppf(yi,0.0,std_scaler)]]).astype('float32')[0]
            plt.scatter(z_mu[0], z_mu[1], color="red", s=3)


    # Encode the sampled inputs
    examples_z = gan.encoder(examples)
    plt.subplot(1, 3, 3)
    # Plot the encodings
    plt.scatter(examples_z[:, 0], examples_z[:, 1], s=2, color=np.array(color_vec)[np.argmax(labels, axis=1)])
    # Add as reference the points used to sample the "distribution manifold"
    for i, xi in enumerate(x_values):
        for j, yi in enumerate(y_values):
            z_mu = np.array([[stats.norm.ppf(xi,0.0,std_scaler), stats.norm.ppf(yi,0.0,std_scaler)]]).astype('float32')[0]
            plt.scatter(z_mu[0], z_mu[1], color="black", s=3, alpha=0.1)
    #Add legend
    for i_c in range(labels.shape[1]):
        plt.scatter([], [], c = color_vec[i_c], label = i_c)
    plt.legend(borderpad=0.2, labelspacing=0.3, handletextpad=0.1, borderaxespad=0.2)
    if not (save_name is None):
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close()

    

def visualize_decodings(gan, examples, partition="test", save_name="/tmp/decs.png", show=False, gray=True):
    n_inputs = len(examples)
    # Encode the sampled inputs
    examples_z = gan.encoder(examples)
    # Decode the encoded inputs
    examples_decoded = gan.decoder(examples_z)

    plt.figure(figsize=(20, 4))
    for i in range(n_inputs):
        # display original
        ax = plt.subplot(2, n_inputs, i + 1)
        plt.imshow(examples[i].numpy().reshape(28, 28))
        plt.title("original")
        if gray: plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n_inputs, i + 1 + n_inputs)
        plt.imshow(examples_decoded[i].numpy().reshape(28, 28))
        plt.title("reconstructed")
        if gray: plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if not (save_name is None):
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close()

def visualize_decodings_data(examples, examples_decoded, save_name="/tmp/decs.png", show=False, shape=(28,28)):
    n_inputs = len(examples)
    plt.figure(figsize=(18, 4))
    for i in range(n_inputs):
        # display original
        ax = plt.subplot(2, n_inputs, i + 1)
        plt.imshow(examples[i].numpy().reshape(shape))
        plt.title("original")
        if len(shape)==2: plt.gray() #plot in grayscale for 2D data
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n_inputs, i + 1 + n_inputs)
        plt.imshow(examples_decoded[i].numpy().reshape(shape))
        plt.title("rec")
        if len(shape)==2: plt.gray() #plot in grayscale for 2D data
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    if not (save_name is None):
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close()


def plot_array(images, save_name=None, show=True, titles=None, gray=True, textsize=12):
    plt.figure(figsize=(20, 4))
    n_images = len(images)
    for i in range(n_images):
        ax = plt.subplot(2, n_images, i + 1)
        plt.imshow(images[i])
        if gray:
            plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if not (titles is None): ax.set_title(titles[i], size=textsize)
    if not (save_name is None):
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close()


def visualize_img(img, title=None, fontsize=12, save_name=None, show=True):
    plt.imshow(img)
    plt.gray()
    plt.gcf().set_size_inches(1.5,1.5)
    plt.axis("off")
    if title is not None:
        plt.title(title, fontsize=fontsize)
    if not (save_name is None):
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show() if show else plt.close()


def plot_prototype_space_gauss(prot_distrib, encoded_examples, data_color, color_vec, labels, cov_matrices,
                               data_alpha=0.05, title="", figsize=(5,3), savefile=None, show=True):
    #Plot the latent space and the location of the prototypes
    cur_proto_set = prot_distrib.mode().numpy()
    plt.scatter(encoded_examples[:,0], encoded_examples[:,1], s=2, color=data_color, alpha=data_alpha)
    plt.scatter(cur_proto_set[:,0], cur_proto_set[:,1], s=20, marker="s", color=np.array(color_vec))
    n_classes = len(labels)

    for i_c in range(n_classes):
        covmat = cov_matrices[i_c]
        class_i_proto = np.copy(cur_proto_set[i_c])
        t = np.linspace(0, np.pi*2, 100)
        for i_scale in np.linspace(1.0, 2.0, 3):  ##NEW
            circle_points = i_scale * np.array([np.cos(t), np.sin(t)]).T
            circle_points = np.matmul(covmat, circle_points.T).T + class_i_proto
            plt.plot(circle_points[:,0], circle_points[:,1], linewidth=1, color=color_vec[i_c], alpha=0.6)
    #x1_min, x2_min = np.min(encoded_examples, axis=0) - 0.5
    #x1_max, x2_max = np.max(encoded_examples, axis=0) + 0.5
    #plt.xlim(x1_min, x1_max)
    #plt.ylim(x2_min, x2_max)
    
    for i_c in range(n_classes):
        plt.scatter([], [], c = color_vec[i_c], label = labels[i_c])
    plt.legend(borderpad=0.2, labelspacing=0.3, handletextpad=0.1, borderaxespad=0.2)
    plt.title(title, fontsize=14)
    plt.gcf().set_size_inches(figsize)
    if not (savefile is None):
        plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close()
        
def get_density_values(density, X1, X2):
    X = np.hstack([X1.flatten()[:, np.newaxis], X2.flatten()[:, np.newaxis]])[:, np.newaxis, :]
    density_values  = density(X).numpy()
    return density_values

def plot_density_contours(density_values, X1, X2, contour_kwargs, ax = None):
    ax.contour(X1, X2, density_values, **contour_kwargs)
    return(ax)


def plot_prototype_space_mixture(prot_distrib, encoded_examples, data_color, color_vec, labels, data_alpha=0.05,
                                 savefile = None, show=True, outfig=False,
                                 title="", figsize=(4,4)):
    #Plot the latent space and the location of the prototypes
    fig, ax = plt.subplots(figsize = figsize)
    ax.scatter(encoded_examples[:,0], encoded_examples[:,1], s=2, color=data_color, alpha=data_alpha)
    
    x1_min, x2_min = np.min(encoded_examples, axis=0) - 0.5
    x1_max, x2_max = np.max(encoded_examples, axis=0) + 0.5
    x1 = np.linspace(x1_min, x1_max, 1000)
    x2 = np.linspace(x2_min, x2_max, 1000)
    X1, X2 = np.meshgrid(x1, x2)
    #contour_levels = np.linspace(0.01, 2, 5) #(1e-2, 10 ** (-0.8), 5) 
    contour_levels  = 4
    
    q_densities = get_density_values(prot_distrib.prob, X1, X2)

    n_classes = len(labels)

    for i_c in range(n_classes):
        ax = plot_density_contours(q_densities[:,i_c].reshape(X1.shape), X1, X2,
                                   {'levels': contour_levels, 'colors': color_vec[i_c], 'alpha': 0.6}, ax = ax)
    for i_c in range(n_classes):
        plt.scatter([], [], c = color_vec[i_c], label = labels[i_c])
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(title, fontsize=14)
    plt.legend(borderpad=0.2, labelspacing=0.3, handletextpad=0.1, borderaxespad=0.2)
    plt.gcf().set_size_inches(figsize)
    if not (savefile is None):
        plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close()
    if outfig: return fig, ax
    



def plot_prototype_grid(nx, n_classes, model, input_shape_full, labels_str, size, savefile, show):
    #Sample several prototypes per class
    ny = n_classes
    n_channels = len(input_shape_full)
    if(n_channels==3):
        is0, is1, is2 = input_shape_full #3 channels
        canvas = np.empty((is1 * ny, is0 * nx, is2))
    else:
        is0, is1 = input_shape_full #2 channels
        canvas = np.empty((is1 * ny, is0 * nx))

    #Sample and include the prototypes
    for i in range(nx):
        decoded_prototypes = model.decoder(model.sample_prototypes(1)[0]).numpy()
        for j in range(ny):
            if(n_channels==3): #3 channels
                unnorm_protos = dataset_handler.unnormalize_data(decoded_prototypes[j].reshape(input_shape_full))
                canvas[j*is1:(j+1)*is1, (nx-i-1) * is0:(nx-i)*is0, :] = unnorm_protos
            else: #2 channels
                canvas[j*is1:(j+1)*is1, (nx-i-1) * is0:(nx-i)*is0] = decoded_prototypes[j].reshape(input_shape_full)
    cmap = "gray" if n_channels==2 else None
    plt.imshow(canvas, origin="upper", cmap=cmap)
    plt.yticks(np.arange(0, is1 * ny, is1)+is1//2, ["%s"%(s) for s in labels_str], fontsize=10)
    [plt.axvline(is0*i, color="grey", linewidth=0.5) for i in range(nx)]
    plt.gca().tick_params(axis=u'y', which=u'major',length=0)
    plt.xticks(np.arange(0, is0 * nx, is0)+is0//2, ["$R_{%d}$"%(i+1) for i in range(nx)]) 
    #plt.axis("off")
    plt.tight_layout()
    plt.gcf().set_size_inches(size)
    plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close()


def plot_grid(sel_imgs, nx, ny, input_shape_full, savefile=None, size=(3,3), show=True):
    #nx, ny = 2,3
    n_channels = len(input_shape_full)
    if(n_channels==3):
        is0, is1, is2 = input_shape_full #3 channels
        canvas2 = np.empty((is1 * ny, is0 * nx, is2))
    else:
        is0, is1 = input_shape_full #2 channels
        canvas2 = np.empty((is1 * ny, is0 * nx))
    
    cont = 0
    for j in range(ny):
        for i in reversed(range(nx)):
            sel_img = sel_imgs[cont]
            if(n_channels==3): #3 channels
                canvas2[j*is1:(j+1)*is1, (nx-i-1) * is0:(nx-i)*is0, :] = sel_img
            else: #2 channels
                canvas2[j*is1:(j+1)*is1, (nx-i-1) * is0:(nx-i)*is0] = sel_img
            cont+=1
    cmap = "gray" if n_channels==2 else None
    plt.imshow(canvas2, origin="upper", cmap=cmap)
    #plt.yticks(np.arange(0, is1 * ny, is1)+is1//2, ["%s"%(s) for s in labels_str], fontsize=10)
    [plt.axvline(is0*i, color="#000000", linewidth=0.5) for i in range(1,nx)]
    [plt.axhline(is1*i, color="#000000", linewidth=0.5) for i in range(1,ny)]
    plt.gca().tick_params(axis=u'y', which=u'major',length=0)
    #plt.xticks(np.arange(, is0 * nx, is0)+is0//2, ["$R_{%d}$"%(i+1) for i in range(nx)]) 
    plt.axis("off")
    plt.tight_layout()
    plt.gcf().set_size_inches(size)
    if savefile is not None: plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close()


def plot_latent_space_TSNE(reduced_examples, data_color_tsne, labels, color_vec, savefile=None, size=(6,3), show=False):
    plt.scatter(reduced_examples[:, 0], reduced_examples[:, 1], color=data_color_tsne, s=2, alpha=0.1) 
    for i_c in range(len(labels)): plt.scatter([], [], c = color_vec[i_c], label = labels[i_c])
    plt.legend()
    plt.gcf().set_size_inches(size)
    if not (savefile is None):  plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close()


def plot_latent_space_SENN(encoded_examples, data_color, prototypes, labels, color_vec, savefile=None, size=(6,3), show=False):
    plt.scatter(encoded_examples[:, 0], encoded_examples[:, 1], color=data_color, s=2, alpha=0.1) 
    plt.scatter(prototypes[:, 0], prototypes[:, 1], color="black", s=10, alpha=0.9) 
    for i_c in range(len(labels)): plt.scatter([], [], c = color_vec[i_c], label = labels[i_c])
    plt.legend()
    plt.gcf().set_size_inches(size)
    if not (savefile is None):  plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.show() if show else plt.close()



def plot_expl_diagonal(images_to_plot, info_below=None,
                       margin_h=28, margin_v=5, basic_dim=28, space_last_h=0, space_last_v=0,
                       guide_color="white", border_color=None, border_offset=3,
                       save_name=None, show=True):
    
    num_imgs_plot = len(images_to_plot)
    assert num_imgs_plot is None or len(info_below) == num_imgs_plot

    big_img_v = basic_dim + (margin_v * (num_imgs_plot - 1) + space_last_v)
    big_img_h = basic_dim + (margin_h * (num_imgs_plot - 1) + space_last_h)
    big_img_size = (big_img_v, big_img_h)

    big_img = np.ones(big_img_size)

    curh1, curh2 = big_img_h - basic_dim, big_img_h
    curv1, curv2 = 0, basic_dim

    for i in range(num_imgs_plot):
        # Select the current image
        cur_img = np.copy(images_to_plot[i])
        #cur_img = add_border_to_img(cur_img, 0.1, attenuation=0.5) #Optional: add borders to the margins

        # Add image to the plot
        big_img[curv1:curv2, curh1:curh2] = np.copy(cur_img)
        # Update indices
        if i == 0:
            curv1, curv2 = curv1 + space_last_v, curv2 + space_last_v
            curh1, curh2 = curh1 - space_last_h, curh2 - space_last_h
        curv1, curv2 = curv1 + margin_v, curv2 + margin_v
        curh1, curh2 = curh1 - margin_h, curh2 - margin_h

    # Display the images
    plt.imshow(big_img, cmap="gray")

    # Add some guidelines
    if not (guide_color is None):
        n_points = 2
        xx = np.linspace(0, big_img_h - basic_dim, num=n_points) - 0.2
        yy = np.linspace(0, big_img_v - basic_dim, num=n_points)[::-1] - 0.2
        plt.plot(xx, yy, linestyle="--", color=guide_color, alpha=0.5)
        plt.plot(xx + basic_dim - 0.5, yy, linestyle="--", color=guide_color, alpha=0.4)
        plt.plot(xx + basic_dim - 0.5, yy + basic_dim - 0.5, linestyle="--", color=guide_color, alpha=0.5)
        # plt.fill_between(xx, yy, yy+basic_dim, color=color_vec[0], alpha=0.1)

    # Add some guidelines
    if not (border_color is None):
        n_points = 2
        bo = border_offset
        for xx in [0,big_img_h]:
            plt.plot([xx,xx], [0,big_img_v+bo], linestyle="-", color=border_color, linewidth=3, alpha=1)
        for yy in [0,big_img_v+bo]:
            plt.plot([0,big_img_h], [yy,yy], linestyle="-", color=border_color, linewidth=3, alpha=1)

        # offmin = 4
        # xx = 0
        # plt.plot([xx-offmin,xx-offmin], [0,big_img_v+bo], linestyle="-", color=border_color, linewidth=3, alpha=0.9)
        # xx = big_img_h
        # plt.plot([xx+offmin,xx+offmin], [0,big_img_v+bo], linestyle="-", color=border_color, linewidth=3, alpha=0.9)
        # yy = 0
        # plt.plot([0-offmin,big_img_h+offmin], [yy-offmin,yy-offmin], linestyle="-", color=border_color, linewidth=3, alpha=0.9)
        # yy = big_img_v + b0
        # plt.plot([0-offmin,big_img_h+offmin], [yy+offmin,yy+offmin], linestyle="-", color=border_color, linewidth=3, alpha=0.9)

    #Add dots before the last image #TODO: very ad-hoc, configure it better!
    plt.plot(big_img_h-basic_dim-2, big_img_v/2, 'o', color="black", markersize=2)
    plt.plot(big_img_h-basic_dim-5, big_img_v/2, 'o', color="black", markersize=2)
    plt.plot(big_img_h-basic_dim-8, big_img_v/2, 'o', color="black", markersize=2)

    plt.axis("off")

    # Plot text
    curh = big_img_h - basic_dim // 2 - 10
    curv = basic_dim + 7
    for i in range(num_imgs_plot):
        plt.text(curh, curv, info_below[i], size=20)
        # Update indices
        if i == 0:
            curv = curv + space_last_v
            curh = curh - space_last_h
        curv = curv + margin_v
        curh = curh - margin_h

    if not (save_name is None):
        plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=True)
    plt.show() if show else plt.close()



def add_border_to_img(img, color, attenuation=None):
    if attenuation is None:
        img[0,:] = color;   img[-1,:] = color
        img[:,0] = color;   img[:,-1] = color
    else:
        img_bu = np.copy(img)
        img[0,:]  = img_bu[0,:] *attenuation
        img[-1,:] = img_bu[-1,:]*attenuation
        img[:,0]  = img_bu[:,0] *attenuation
        img[:,-1] = img_bu[:,-1]*attenuation
    return img

