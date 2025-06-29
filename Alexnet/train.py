from tqdm import tqdm
import torch




def train_script(model,optim,criteration,trainDataLoader,testDataLoader,device='cpu',epochs=100):

    trainLosslist = []
    testLosslist =[]
    trainLoss = 0
    testLoss = 0
    for i in tqdm(range(epochs)):
        model.train()
        for x, y in trainDataLoader:
            x,y = x.to(device),y.to(device)
            
            optim.zero_grad()
            train_pred = model(x)
            loss = criteration(train_pred,y)

            loss.backward()

            optim.step()

            trainLoss += loss
            
        model.eval()
        with torch.inference_mode():
            for test_x,test_y in testDataLoader:
                test_x,test_y = x.to(device), y.to(device)
                test_pred = model(x)
                test_loss = criteration(test_pred,y)

                testLoss += loss

        print(f'Train Loss after iteration {i} : {trainLoss/len(trainDataLoader.dataset)}')
        print(f'Test Loss occured after iteration {i} : {testLoss/len(testDataLoader.dataset)}')
        

        trainLosslist.append(trainLoss/len(trainDataLoader.dataset))
        testLosslist.append(testLoss/len(testDataLoader.dataset))
