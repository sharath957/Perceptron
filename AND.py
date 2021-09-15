from utils.model import perceptron
from utils.all_utils import prepare_data,save_plot,save_model
import pandas as pd 

AND = {
    'x1':[0,0,1,1],
    'x2':[0,1,0,1],
    'y':[0,0,0,1],
}

df = pd.DataFrame(AND)
print(df) 

X,y = prepare_data(df)

ETA = 0.3 # 0 and 1
EPOCHS = 10

model = perceptron(eta=ETA, epochs=EPOCHS)
model.fit(X, y)

_ = model.total_loss()

save_model(model,filename="AND.model")
save_plot(df,"AND.png",model)