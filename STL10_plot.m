clear all;
close all;
data=load('/home/lili/Workspace/store_weights/MegaNet-tf/STL10.txt')

data1=data(:,1)*100
resnet=data(:,2)*100
hamiltonian=data(:,3)*100

plot(data1,hamiltonian,'r-o','LineWidth',2)
hold on;
plot(data1,resnet,'b--s',...
    'LineWidth',2)



h=legend('HRNet','ResNet')
xlabel('Training Data Percentage (%)')
ylabel('Accuracy (%)')
set(h,'FontSize',14)
set(gca,'FontSize',12)
grid on
xlim([5 70])
