clear all
close all 
clc 

% S = webread('http://api.wunderground.com/api/a0eb2000d73ea790/history_20150519/q/FL/Boca_Raton.json');
% 
% 
% 
% A=S.history.observations
% 
% A(2).date
% 
% A(2).tempm

% S = webread('http://api.wunderground.com/api/a0eb2000d73ea790/animatedradar/q/FL/Boca_Raton.png?width=800&height=800&newmaps=0');
% S = webread('http://api.wunderground.com/api/a0eb2000d73ea790/radar/q/FL/Boca_Raton.png?width=800&height=800&newmaps=0');
% S = webread('http://api.wunderground.com/api/a0eb2000d73ea790/satellite/q/FL/Boca_Raton.gif?width=280&height=280&basemap=1');


S = webread('http://api.wunderground.com/api/a0eb2000d73ea790/animatedradar/animatedsatellite/history_20150517/q/FL/Boca_Raton.gif?num=8&delay=50&interval=30');

for i =1:8
imagesc(S(:,:,1,i))
pause
end

% for i =1:7
% imagesc(S(:,:,1,i+1)-S(:,:,1,i))
% pause
% end

