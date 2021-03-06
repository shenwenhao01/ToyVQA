# paths
q_path = 'data/questions' # directory of questions
a_path = 'data/annotations' # directory of annotations
train_path = 'data/images/train'  # directory of training images
val_path = 'data/images/val'  # directory of validation images
test_path = 'data/images/test'  # directory of test images
preprocessed_path = './resnet-14x14.h5'  # path where preprocessed features are saved to and loaded from
preprocessed_path_test = './resnet-14x14-test.h5'  # test set preprocessed features
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to

task = 'OpenEnded'
dataset = 'mscoco'

# preprocess config
preprocess_batch_size = 10
image_size = 448  # scale shorter end of image to this size and centre crop
output_size = image_size // 32  # size of the feature maps after processing through a network
output_features = 2048  # number of feature maps thereof
central_fraction = 0.875  # only take this much of the centre when scaling and centre cropping

# training config
pretrained = False # set to true to continue training with a checkpoint or a pretrained model
pretrained_model_path = 'logs/2021-07-10_13:44:34.pth' # your pretrained model path
epochs = 30
batch_size = 64
#initial_lr = 2e-3  # default Adam lr
initial_lr = 1e-3
lr_halflife = 50000  # in iterations
data_workers = 1
max_answers = 350

# context seleciton
device = 'GPU'
# device = 'Ascend'

lr_decay_step = 2
save_interval = 5