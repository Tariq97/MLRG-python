function contour_plot( x, y, t, title_txt )

hc = contour(x,y,t,40,'LineWidth',1);
xlabel('x');
ylabel('z');
set(gca,'ydir','reverse');
axis equal;
axis tight;
colormap('jet')
colorbar;
title(title_txt);

end

