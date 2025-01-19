clear all;
close all;

load('test_MM1.mat');

if test.DzDx(1) == test.DzDx(2)
    h = test.DzDx(1);
else
    disp('raytracer only supports Dx = Dz'); 
    return
end

n(1) = test.NzNx(2);
n(2) = test.NzNx(1);
                         % plots in the screenshot indicate 
s(1) = test.hypoNzNx(1); % that hypoNzNx(1) is x-component 
s(2) = test.hypoNzNx(2); % and  hypoNzNx(2) is z-component

X = ([1:n(1)]-1)*h;
Z = ([1:n(2)]-1)*h;

[x,z] = meshgrid(X,Z);

figure('Color','w');
set(gcf,'Units','centimeters')
pos = get(gcf,'Position'); 
xSize = 40; ySize = 15;
x0 = (2*pos(1) + pos(3))*0.5;
y0 = (2*pos(2) + pos(4))*0.5;
set(gcf,'Position',[x0-xSize*0.5 y0-ySize*0.5 xSize ySize]) 

rupvel = test.vr_hom';
t = rupvel2onsettime(rupvel,n,h,s);


%%

figure;

subplot(2,2,1);
contour_plot(x,z,t,'hom');

subplot(2,2,2);
contour_plot_b(x,z,t);
hold on;

rupvel = test.vr_grd';
t = rupvel2onsettime(rupvel,n,h,s);

subplot(2,2,2);
contour_plot(x,z,t,'grd');

rupvel = test.vr_het';
t = rupvel2onsettime(rupvel,n,h,s);

subplot(2,2,3);
contour_plot(x,z,t,'het');

rupvel = test.vr_svr';
t = rupvel2onsettime(rupvel,n,h,s);

subplot(2,2,4);
contour_plot(x,z,t,'svr');
