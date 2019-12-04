classdef VS < dagnn.Layer
    % @author: Ran Song
    properties
        vstride = 24;%standard setup is 24
        vfchannel=4096;
    end
    
    methods
        
        function self = VS(varargin)
            self.load(varargin) ;
            self.vstride = self.vstride;
            self.vfchannel= self.vfchannel;
        end
        
        function outputs = forward(self, inputs, params )
            % -------------------------------------------------------------------------
            [sz1, sz2, sz3, sz4] = size(inputs{1});
            if mod(sz4,self.vstride)~=0 ,
                error('All shapes should have same number of views.');
            end
            
            if(sz1*sz2~=1)
                error ('The first 2 dimentions of the tensor should be 1x1.')
            end
%             view_tensor = reshape(inputs{1},[sz1 sz2 sz3 self.vstride sz4/self.vstride]);
%            dist=sz3*self.vstride;
dist=sz3*sz4;
            inputfeats=reshape(inputs{1},[sz3 self.vstride sz4/self.vstride]);
            sa_tensor=zeros(self.vstride,sz4/self.vstride);
            for k=1:sz4/self.vstride
                for i=1:self.vstride
                    diffe=0;
                     for j=1:self.vstride
                         if i~=j   
                          difference=norm(inputfeats(:,i,k)-inputfeats(:,j,k));
                          diffe=diffe+difference;
                         end  
                      end
%                 difference=repmat(reshape(view_tensor(:,:,:,i,k),[sz3 1]),[1 self.vstride])-reshape(view_tensor(:,:,:,:,k),[sz3 self.vstride]);
%                 difference=difference(:);
%                 sa_tensor(i,k)=sqrt(sum(difference.^2));%/dist;
                  sa_tensor(i,k)=diffe;
                end
%                 sa_tensor(:,k)=sa_tensor(:,k);%/sum(sa_tensor(:,k));
            end
%              for k=1:sz4/self.vstride
%                 for i=1:self.vstride
%                     diffe=norm(inputfeats(:,i,k));
% %                 difference=repmat(reshape(view_tensor(:,:,:,i,k),[sz3 1]),[1 self.vstride])-reshape(view_tensor(:,:,:,:,k),[sz3 self.vstride]);
% %                 difference=difference(:);
% %                 sa_tensor(i,k)=sqrt(sum(difference.^2));%/dist;
%                   sa_tensor(i,k)=diffe;
%                 end
% %                 sa_tensor(:,k)=sa_tensor(:,k);%/sum(sa_tensor(:,k));
%             end
            
            
%             distinctmat=zeros(self.vstride,self.vstride,sz4/self.vstride);
%             
%             for k=1:sz4/self.vstride
%                 for i=1:self.vstride
%                     for j=1:self.vstride
%                         view_feat1=view_tensor(:,:,:,i,k);
%                         view_feat2=view_tensor(:,:,:,j,k);
%                         distinctmat(i,j,k)=sqrt(sum(   (view_feat1(:)-view_feat2(:))  .^2));
%                     end
%                 end
%             end
%             
%             sa_tensor=zeros(self.vstride,sz4/self.vstride);
%             for k=1:sz4/self.vstride
%                 [eigf,~]=eigs(distinctmat(:,:,k),1,'la');
%                 eigf=eigf-min(eigf);
%                 sa_tensor(:,k)=eigf;
%             end
            outputs{1} = reshape(sa_tensor/dist,[sz1 sz2 self.vstride sz4/self.vstride]);   
        end
        
         
        % -------------------------------------------------------------------------
        
        function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
            % -------------------------------------------------------------------------
            [sz1, sz2, sz3, sz4] = size(inputs{1});
            inputfeats=reshape(inputs{1},[sz3 self.vstride sz4/self.vstride]);
%              dist=sz3*self.vstride;
dist=sz3*sz4;
%             [sz1, sz2, sz3, sz4] = size(derOutputs{1}); %1x 1x self.vstride x numofobjects
           numshape=sz4/self.vstride;
%             y=derOutputs{1}/numshape;
%             a=zeros(sz1,sz2,sz3,self.vstride,numshape,'single');
%             for i=1:numshape
%                 for j=1:self.vstride
%                 a(:,:,:,j,i)=2*reshape(y(:,:,j,i),[1 1])*inputfeats(:,:,:,j,i)/dist;
%                 end
%             end
               a=zeros(sz3,self.vstride,numshape);
%                b=a;
              %c=ones(sz3,self.vstride,numshape,'single');
               y=reshape(derOutputs{1},[self.vstride numshape]);
%                for i=1:numshape
%                    for j=1:self.vstride
% %                        for k=1:sz3 
% %                        b(k,j,i)=2*sz3*inputfeats(k,j,i)-2*sum(inputfeats(:,j,i));
% % %                        b(k,j,i)=2*inputfeats(k,j,i);
% %                        end
% %                       a(:,j,i)=b(:,j,i)*y(j,i);
%                        a(:,j,i)=c(:,j,i)*y(j,i);
%                         a(:,j,i)=a(:,j,i)/sum(a(:,j,i));
%                    end
%                end
              
               for i=1:numshape
                   for j=1:self.vstride
                       dydxj=zeros(sz3,1);
                       for k=1:self.vstride
                         if j~=k   
%                              norm(inputfeats(:,j,i)-inputfeats(:,k,i)+eps)
                          dydxjk=(inputfeats(:,j,i)-inputfeats(:,k,i))/norm(inputfeats(:,j,i)-inputfeats(:,k,i)+eps);
                          dydxj=dydxj+dydxjk;
                         end  
                       end
                       
                        a(:,j,i)=y(j,i)*dydxj;
                   end
               end
                
%                for i=1:numshape
%                    for j=1:self.vstride
%                        
%                        dydxj=inputfeats(:,j,i)/norm(inputfeats(:,j,i));
%                        
%                         a(:,j,i)=y(j,i)*dydxj;
%                    end
%                end
            
           
             %derOutputs{1}=reshape(repmat(  reshape(derOutputs{1} / (self.vstride*sum(inputfeats(:))),[sz1 sz2 1 sz3 sz4]) ,[1 1 self.vfchannel 1 1]),[sz1 sz2 self.vfchannel sz3 sz4]);
          derInputs{1}=reshape(a/dist,[sz1 sz2 sz3 sz4]);
            derParams={};    
%                         %validate derivative
%                         dererror=zeros(1,100);
%                         dzdy=derOutputs{1};
%                         dzdx=derInputs{1};
%                        for iii=1:100
%                         ex = randn(size(inputs{1}), 'single') ;
%             
%                         eta = 0.0001 ;
%                         xp = inputs{1} + eta * ex  ;
%             
%             
%                         outputs = forward(self, inputs, params);
%                         yo=outputs{1};
%                         inputs2=inputs;
%                         inputs2{1}=xp;
%             
%                         outputsp = forward(self, inputs2, params);
%                         yp=outputsp{1};
%                         dzdx_empirical = sum(dzdy(:) .* (yp(:) - yo(:)) / eta) ;
%                         dzdx_computed= sum(dzdx(:) .* ex(:)) ;
%             
%                        dd=abs(1 - dzdx_empirical/dzdx_computed)*100;
%                   
%                       dererror(iii)=dd;
%                     
%                        end
%                        mean(dererror)
                      
%             
%             
%             if strcmp(self.method, 'avg'),
%                 derInputs{1} = ...
%                     reshape(repmat(reshape(derOutputs{1} / self.vstride, ...
%                     [sz1 sz2 sz3 1 sz4]), ...
%                     [1 1 1 self.vstride 1]),...
%                     [sz1 sz2 sz3 self.vstride*sz4]);
%             elseif strcmp(self.method, 'max'),
%                 [~,I] = max(reshape(permute(inputs{1},[4 1 2 3]), ...
%                     [self.vstride, sz4*sz1*sz2*sz3]),[],1);
%                 Ind = zeros(self.vstride,sz4*sz1*sz2*sz3, 'single');
%                 Ind(sub2ind(size(Ind),I,1:length(I))) = 1;
%                 Ind = permute(reshape(Ind,[self.vstride*sz4,sz1,sz2,sz3]),[2 3 4 1]);
%                 derInputs{1} = ...
%                     reshape(repmat(reshape(derOutputs{1}, ...
%                     [sz1 sz2 sz3 1 sz4]), ...
%                     [1 1 1 self.vstride 1]),...
%                     [sz1 sz2 sz3 self.vstride*sz4]) .* Ind;
%             elseif strcmp(layer.method, 'cat'),
%                 derInputs{1} = reshape(derOutputs{1}, [sz1 sz2 sz3/self.vstride sz4*self.vstride]);
%             else
%                 error('Unknown viewpool method: %s', self.method);
%             end
%             derParams={};
        end
    end
    
end


