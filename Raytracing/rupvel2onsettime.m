function T = rupvel2onsettime(rupvel,n,h,s)

%==========================================================================
%==========================================================================
%==========================================================================
% n = [50, 40, 60];
% h = 100.;
% s = [10, 30, 40];
% 
% % test: read existing file and process it
% % read onset times from binary file:
% fid=fopen('Vs_sample.bin','rb');
% t = fread(fid,n(1)*n(2)*n(3),'single');
% fclose(fid);
% 
% for k = 1:n(3)
%    for j = 1:n(2)
%       rec = ( ((k-1) * n(2) + j) - 1)*n(1) + 1;
%       rupvel(:,j,k) = t(rec:rec+n(1)-1);
%    end
% end
%==========================================================================
%==========================================================================
%==========================================================================

% extend input for 3d - raytracer supports only 3D input...
n(3) = 1;
s(3) = 1;

% convert and write rupvel to file in format for raytracer
for j = 1:n(2)
  rec = (j - 1)*n(1) + 1;
  buffer(rec:rec+n(1)-1) = rupvel(:,j,1);
end


fid=fopen('rupvel.bin','wb');
fwrite(fid,buffer,'single');
fclose(fid);

% prepare input file for raytracer
fid = fopen('input.inp','w');
fprintf(fid,'rupvel.bin \n');
fprintf(fid,'%f \n',h);
fprintf(fid,'%i \n',n(1));
fprintf(fid,'%i \n',n(2));
fprintf(fid,'%i \n',n(3));
fprintf(fid,'1 \n'); % only one 'source'
fprintf(fid,'%i \n',s(1));
fprintf(fid,'%i \n',s(2));
fprintf(fid,'%i \n',s(3));
fclose(fid);

% run the raytracer
ret = system('./raytracer.exe');
if ret~=0; disp('Error running raytracer...'); end

% read raytracer results and convert to output format
fid=fopen('first_arrival.out','rb');
t = fread(fid,n(1)*n(2)*n(3),'single');
fclose(fid);

for j = 1:n(2)
  rec = ( j - 1)*n(1) + 1;
  T(j,:) = t(rec:rec+n(1)-1);
end


