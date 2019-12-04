# CFSCNN
The source codes are based on matconvnet.

Download the pretrained net from
https://drive.google.com/open?id=1yc982kKOCVwr5OROoMCbx_qqGwLe_Wv4

Save the downloaded net file 'net-deployed.mat' in \data\deploynet\
For a demo, simply implement the following lines in MATLAB command window

setup;
sa1 = cnn_output_fast('skateboard.off');

You can also input a scene:

sa2 = cnn_output_fast('human.off');

Note that the program accepts OFF or OBJ files for the input 3D model.

If you want to train it using the modelnet40 dataset, please follow the steps below:
1. Download the baseline VGG19 net from the official website of matconvnet or
https://drive.google.com/open?id=1jezzXXTUnySn0tTtpt41lP61o8r1c71N

Save it in \data\models\

2. Download the ModelNet40 datasets and render all of the models using render_views_of_all_meshes_in_a_folder.m and create_data.m in the utils folder. Then save the rendered images (including 40 folders) in \data\modelnet40.

3. Run run_train.m where the get_imdb function only need to be implemented at the first time of training.

Please cite our paper if you use the codes:

Ran Song, Yonghuai Liu, Paul L. Rosin. Mesh Saliency via Weakly Supervised Classification-for-Saliency CNN. IEEE Transactions on Visualization and Computer Graphics, 13 pages, 2019.
