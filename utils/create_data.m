listing=rdir(modelnet40path);%add your modelnet40 dataset path
for i=1:numel(listing)
   
   trainset = fullfile(listing(i).name,'train');
   testset = fullfile(listing(i).name,'test');
   a=dir([trainset '/*.jpg']);
   if size(a,1)==2400
       
       fprintf(['folder ' num2str(i) ' is rendered']);
       continue;
   end
   render_views_of_all_meshes_in_a_folder(trainset);
   render_views_of_all_meshes_in_a_folder(testset);
   delete(fullfile(trainset,'*.off'));
   delete(fullfile(testset,'*.off'));
   
   copyfile(fullfile(testset,'*.jpg'),trainset);
   delete(fullfile(testset,'*.jpg'));
   rmdir(testset);
end