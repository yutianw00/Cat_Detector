import model
import numpy as np
from PIL import Image

# convert an image to a tensor for model input
def image_to_input(img):
    '''
    @param img: numpy array of dim (3, 32, 32)
    @return res: torch tensor ready to pass into NN
    '''
    # x y z -> y z x
    res = np.transpose(img, (2, 0, 1))
    res = torch.from_numpy(res)
    res = (res/256 - 0.5) * 2 
    res = torch.unsqueeze(res, 0)
    return res

# from path of an image, convert it to numpy array
def resize_img(PATH):
    im = Image.open(PATH)  
    im = im.resize((32, 32)) 
    im = np.array(im)
    return im

def get_prediction(outputs, threshold=0.2):
    # the smaller the threshold, more easier to recognize a cat, but 
    # also more likely to recognize non-cat objects as cats
    p = outputs>threshold
    p = p.reshape(-1, 1)
    return p.int()

def test_img(PATH, net):
    img_input = image_to_input(resize_img(PATH))
    output = net(img_input)
    predicted = get_prediction(output)
    if (predicted.item() == 0):
        return "Doesn't look like a cat"
    else:
        return "This looks like a cat"

def load_env(PATH = "cifar_resnet_binary08.pth"):
    net = model.myNet(model.ResBlock)
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))




