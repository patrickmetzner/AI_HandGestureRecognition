import os
import random
import glob
import shutil

# Organize data into trainingImages, validationImages, testImages directories
os.chdir('images')
if os.path.isdir('trainingImages/thumbsUp') is False:
    os.makedirs('trainingImages/thumbsUp')
    os.makedirs('trainingImages/thumbsDown')
    os.makedirs('validationImages/thumbsUp')
    os.makedirs('validationImages/thumbsDown')
    os.makedirs('testImages/thumbsUp')
    os.makedirs('testImages/thumbsDown')

for c in random.sample(glob.glob('thumbsUp*'), 1000):
    shutil.move(c, 'trainingImages/thumbsUp')
for c in random.sample(glob.glob('thumbsDown*'), 1000):
    shutil.move(c, 'trainingImages/thumbsDown')
for c in random.sample(glob.glob('thumbsUp*'), 200):
    shutil.move(c, 'validationImages/thumbsUp')
for c in random.sample(glob.glob('thumbsDown*'), 200):
    shutil.move(c, 'validationImages/thumbsDown')
for c in random.sample(glob.glob('thumbsUp*'), 20):
    shutil.move(c, 'testImages/thumbsUp')
for c in random.sample(glob.glob('thumbsDown*'), 20):
    shutil.move(c, 'testImages/thumbsDown')

os.chdir('../')

# NN name?

# Run next program? os.system("Image_Text.py")
