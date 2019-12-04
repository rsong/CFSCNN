
function sa = cnn_output_fast( path_to_shape)
%
%   path_to_shape:: 
%        can be either a filename for a mesh in OBJ/OFF format
%        or a name of folder containing multiple OBJ/OFF meshes
%   `cnnModel`:: (default) ''
%       this is a matlab file with the saved CNN parameters
%   `applyMetric`:: (default) false
%       set to true to disable transforming descriptor based on specified
%       distance metric
%   `metricModel`:: (default:) ''
%       this is a matlab file with the saved metric parameters
%       if the default file is not found, it will attempt to download from our
%       server
%   `gpus`:: (default) []
%       set to use GPU

outputSize = 224;

mesh_filenames(1).name = path_to_shape;


modelPath = '.\data\deploynet\net-deployed.mat' ;

net=load(modelPath') ;
net = dagnn.DagNN.loadobj(net) ;

net.mode = 'test' ;
predVar = net.getVarIndex('prob') ;
predVar_im = net.getVarIndex('input') ;
predVar_sa = net.getVarIndex('xViewSa') ;

nChannels = size(net.params(1).value,3);

averageImage = net.meta.normalization.averageImage;
if numel(averageImage)==nChannels,
    averageImage = reshape(averageImage, [1 1 nChannels]);
end

fig = figure('Visible','off');
for i=1:length(mesh_filenames)
    mesh = loadMesh( mesh_filenames(i).name );
    xn1=max(mesh.V(1,:));
    xn2=min(mesh.V(1,:));
    yn1=max(mesh.V(2,:));
    yn2=min(mesh.V(2,:));
    zn1=max(mesh.V(3,:));
    zn2=min(mesh.V(3,:));
    bbox=sqrt((xn1-xn2).^2+(yn1-yn2).^2+(zn1-zn2).^2);
    
    mesh.V(1,:)=mesh.V(1,:)-0.5*(xn1+xn2);
    mesh.V(2,:)=mesh.V(2,:)-0.5*(yn1+yn2);
    mesh.V(3,:)=mesh.V(3,:)-0.5*(zn1+zn2);
    
    [nViews,ims,viewpoints,cam_angles] = render_views(mesh,'figHandle', fig);
  
    imsp=cell(1, nViews);
    
    if length(mesh.F)>4000
        [p,t] = perform_mesh_simplification(mesh.V',mesh.F',2500);
        knng=knnsearch(p,mesh.V');
    else
        p=mesh.V';
        t=mesh.F';
        knng=1:length(mesh.V);
    end
   
    W=density(t,p);
    imsvsas=zeros(length(p),nViews)+0.5;
    
    imsvsa=zeros(length(mesh.V),nViews);
    [a,b,c]=size(ims{1});
    im=zeros(a,b,c,numel(ims),'single');
  
    for j=1:numel(ims)
        im(:,:,:,j) = ims{j};
    end
    im=single(im);
    
    im_ = bsxfun(@minus, im,averageImage);
    
    inputs = {'input',im_};
    net.vars(predVar_sa).precious=1;
    net.eval(inputs) ;
    scores = gather(net.vars(predVar).value) ;
    [~,ind_class]=max(scores,[],3);
    vweights=gather(net.vars(predVar_sa).value) ;
    vweights=vweights(:);
    
    dzdy=zeros(1,1,40,1);
    dzdy(1,1,ind_class,1)=0.1;
    deout={'prob',dzdy};
    net.vars(predVar_im).precious=1;
    net.eval(inputs,deout) ;
    aaa = gather(net.vars(predVar_im).der) ;
    
    msa= imsvsa;
 
    for ii=1:nViews
        az=viewpoints(ii,1);
        el=viewpoints(ii,2);
        cam_angle=cam_angles(ii); 
        [crop,r1,r2,c1,c2]=autocrop(ims{ii});
       
        aa=size(crop,1);
        bb=size(crop,2);
        longside=max(aa,bb);
        shortside=min(aa,bb);
     
        bbb=aaa(:,:,:,ii);
      
        ccc=max(abs(bbb),[],3);
       
        ddd=(ccc-min(ccc(:)))./(max(ccc(:))-min(ccc(:)));
       
        T=viewmtx(az,el,cam_angle);
        
        x4d=[p';ones(1,length(p))];
        x2d = T*x4d;
        %figure,plot(x2d(1,:),x2d(2,:),'r.');axis equal tight;
        xxa=max(x2d(1,:));
        xxi=min(x2d(1,:));
        yya=max(x2d(2,:));
        yyi=min(x2d(2,:));
        
        xx=xxa-xxi;
        yy=yya-yyi;
        
        longx=max(xx,yy);
        shortx=min(xx,yy);
        
        scale1=longside/longx;
        scale2=shortside/shortx;
        
        sscale=0.5*(scale1+scale2);
        x1=x2d(1,:)*sscale;
        y1=x2d(2,:)*sscale;
        
        x2=x1+0.5*outputSize;
        y2=y1-0.5*outputSize;
        
        cropsa=ddd(r1:r2,c1:c2);
       
        imsp{i}=[x2;y2];
        y2=-y2;
        x2=x2-min(x2);
        y2=y2-min(y2);% y2 is row;
       
        [vpy,vpx,vpz]=sph2cart(pi*viewpoints(ii,1)/180,pi*viewpoints(ii,2)/180,bbox);
        vpy=-vpy;
        visibility_v = mark_visible_vertices(p,t,[vpx,vpy,vpz]);
        visibility_v=perform_mesh_smoothing(t,p,visibility_v);
        %
        visibility_v=perform_mesh_smoothing(t,p,visibility_v);
       
        [impointsx,impointsy]=meshgrid(1:bb,1:aa);
        impoints=[impointsx(:) impointsy(:)];
        
        visible=find(visibility_v~=0);
        invisible=find(visibility_v==0);
      
        vx2=x2(visible);
        vy2=y2(visible);
        x2ddd=[vx2(:) vy2(:)];
        ind_cor = knnsearch(impoints,x2ddd);
        
        
        for jj=1:length(visible)
           
            
            row=impoints(ind_cor(jj),2);
            col=impoints(ind_cor(jj),1);
            
            imsvsas(visible(jj),ii)=exp(1-single(W(visible(jj))))/(exp(1-cropsa(row,col)));
            %         end
            
        end
        imsvsas(invisible,ii)=mean(imsvsas(visible,ii));
       
        imsvsas(:,ii)= perform_mesh_smoothing(t,p,imsvsas(:,ii));
        
        imsvsa(:,ii)=imsvsas(knng,ii);
     
        
       imsvsa(:,ii)= perform_mesh_smoothing(mesh.F',mesh.V',imsvsa(:,ii));

        msa(:,ii)=imsvsa(:,ii).*vweights(ii);
    end
    sa=sum(msa,2);
end
close(fig);
close all;
sa(isnan(sa))=min(sa);
sa=(sa-min(sa))/(max(sa)-min(sa));
figure,trisurf(mesh.F',mesh.V(1,:),mesh.V(2,:),mesh.V(3,:),sa);axis equal tight;axis off;shading interp;view(0,180);lightangle(0,180);lighting gouraud;colormap jet;
end

