temp = load('loss.mat');
loss = temp.loss;
%%
figure('Position', [200, 200, 600, 400]);
color = [1 0 0;0 1 0;0 0 1;0.9290 0.6940 0.1250;1 0 1;0 0 0;0.6350 0.0780 0.1840;0 1 1;0.4660 0.6740 0.1880;0.4940 0.1840 0.5560;0.8500 0.3250 0.0980;0 0.4470 0.7410];
for i=1:12
    plot(loss(i,1:12000),'color',color(i,:),'linewidth',1);
    hold on;
end
% xlim([-500,10000])
% ylim([0.05, 0.1])
xlabel('Iteration #')
ylabel('Loss (MSE)')
legend('Batch 1', 'Batch 2','Batch 3', 'Batch 4','Batch 5', 'Batch 6','Batch 7', 'Batch 8','Batch 9', 'Batch 10','Batch 11', 'Batch 12')
set(gca,'FontSize',12,'FontName','Helvetica','LineWidth', 1);

disp(min(loss, [], 2))
disp(mean(min(loss, [], 2),1))