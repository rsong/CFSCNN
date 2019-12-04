listing=rdir('C:\Program Files\MATLAB\R2015a\work\project4\Debug\mvcnn-master\data\modelnet40v5');
for i=1:numel(listing)
    
    imset = listing(i).name;
    
    allimages=dir([imset '/*.jpg']);
    imnum=size(allimages,1);
    j=1;
    while j<=imnum
        imname=allimages(j).name;
        a=imname(end-6:end-4);
        if str2double(a)>12
            delete(fullfile(imset,imname));
            j=j-1;
            allimages=dir([imset '/*.jpg']);
            imnum=size(allimages,1);
        end
        j=j+1;
    end
    
    
    %    if size(a,1)==2400
    %
    %        fprintf(['folder ' num2str(i) ' is rendered']);
    %        continue;
    %    end  
end