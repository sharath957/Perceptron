# Perceptron

## bash
```bash 
git add . && git commit -m "first commit" && git push origin main 

```
## Used to push the commits to git repository  

## To move the file from other directories to working directory 
cp Research\ notebook/Demo.ipynb .

## Add image
![GitHub Logo](plots/or.png)

## Data Set
x1 | x2 | y
-|-|-|
0|0|0|
0|1|0
1|0|0
1|1|1

![GitHub Logo](https://images.unsplash.com/photo-1615789591457-74a63395c990?ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8ZG9tZXN0aWMlMjBjYXR8ZW58MHx8MHx8&ixlib=rb-1.2.1&w=1000&q=80)  

<img src ='https://images.unsplash.com/photo-1615789591457-74a63395c990?ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8ZG9tZXN0aWMlMjBjYXR8ZW58MHx8MHx8&ixlib=rb-1.2.1&w=1000&q=80' width='400' height='500'>

## python code
``` python 
X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = Perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()

save_model(model,filename="and.model") 
save_plot(df,'and.png',model) 
```