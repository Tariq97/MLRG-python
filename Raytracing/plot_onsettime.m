% plot rupture onset time
clear all
close all

%figure;

% grid step
h = 100.;

% dimensions
nx = 50;
ny = 40;
nz = 60;

% source loc:
sx = 10;
sy = 30;
sz = 40;

% read onset times from binary file:
fid=fopen('first_arrival.out','rb');
t = fread(fid,nx*ny*nz,'single');
fclose(fid);

for k = 1:nz
   for j = 1:ny
      disp([num2str(j) ' - ' num2str(k)]);
      rec = ( ((k-1) * ny + j) - 1)*nx + 1;
      T(:,j,k) = t(rec:rec+nx-1);
   end
end

hc = contourslice(T,sy,sx,sz,40);
set(hc,'LineWidth',1);
xlabel('y');
ylabel('x');
zlabel('z');
axis equal;
axis tight;
set(gca,'Zdir','reverse');
colormap('jet')
colorbar;
view(-50,30);


