function contour_plot_b(x,y,t )

hc = contour(x,y,t,40,':k','LineWidth',0.5);
xlabel('x');
ylabel('z');
set(gca,'ydir','reverse');
axis equal;
axis tight;

end

