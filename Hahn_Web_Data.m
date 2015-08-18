


api = 'http://climatedataapi.worldbank.org/climateweb/rest/v1/';
url = [api 'country/cru/tas/year/USA'];
S = webread(url)

year = [S.year];
data = [S.data];
plot(year,data)
xlabel('Year');
ylabel('Temperature (Celsius)');
title('USA Average Temperatures')
axis tight