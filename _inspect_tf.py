import os, glob
root = r'C:\Users\minak\OneDrive\Desktop\EMOTION_AI\Emotion_Detection\tf_env\Lib\site-packages'
print('exists', os.path.exists(os.path.join(root,'tensorflow')))
print('list', sorted(os.listdir(root))[:80])
print('tf glob', sorted(glob.glob(os.path.join(root,'tensorflow','*')))[:80])
