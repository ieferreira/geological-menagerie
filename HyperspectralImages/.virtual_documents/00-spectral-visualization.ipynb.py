import scipy.io
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('data/Indian_pines_corrected.mat')


type(mat)


hsi_img = mat["indian_pines_corrected"]


hsi_img.shape


plt.imshow(hsi_img[:,:,0])


from ipywidgets import interact

@interact(num= (0,144,10))
def plot_img(num):
    plt.imshow(hsi_img[:,:,num])
    plt.show()


hsi_img[1]


type(mat["indian_pines_corrected"])



@interact(num= (0,144,10))
def plot_img(num):
    plt.imshow(hsi_img[:,:,num],cmap="jet")
    plt.show()


import spectral as sl


view = sl.imshow(hsi_img)


pc = sl.principal_components(hsi_img)


xdata = pc.transform(hsi_img)


w = sl.view_nd(xdata[:,:,:15])



