# AI3603大作业：棋类搜索

第16组的AI大作业，组员为闵奕达，唐乐轩，张治凡。本次大作业的目标为在五子棋对战任务中，使用蒙特卡洛树搜索等方法，在有限的计算资源下尽可能战胜该任务中提供的对手。下面将会介绍本次实验的环境配置和文件构成。

------

### 环境配置

```powershell
python == 3.7
numpy == 1.24.4
pygame == 2.1.0
pytorch == 2.1.0
matplotlib == 3.7.3
```

------

### 文件构成

`output`文件夹中为保存的训练日志以及画图的`jupyter notebook`文件。另外，该文件夹中还有两个模型文件，其中`best.model`为最后的结果，战绩最好但是每一步思考时间比较长，`original.model`为训练初期取得对`self_play=1000`的`pure MCTS`全胜的首个模型，虽然对后续更强`pure MCTS`的战绩不如最优模型，但思考时间较短。

由于最后运行代码的视频（即我们得到的模型与`pure MCTS`对弈的具体结果）过大，因此存放在网盘中，链接为：https://pan.baidu.com/s/1qlTlpx77EpRT9pInML0zUg?pwd=sjtu （需要说明的是，文件名中的数字代表我们的模型`best.model`和不同`self_play`次数的对手下棋的结果视频）。

接下来介绍各个`python`文件的作用。

- `ai_play.py`实现了我们的模型与其他对手对弈的功能，相应对局实现在我们写的UI上。
- `pygame_UI.py`实现了新的下棋UI，利用`pygame`库实现了丰富的功能。
- `game.py`定义了五子棋的规则，并且实现了AI对弈和人与AI对弈的步骤。
- `policy_value_net_pytorch.py`定义了网络结构和单步训练。
- `mcts_pure.py`和`mcts_alphaZero.py`实现了`MCTS`的基本逻辑，并分别实现了`pure MCTS`和`our model`的下棋者。
- `train.py`实现了训练过程中的各种功能。

------

### 实验流程

#### 训练

```powershell
python train.py
```

其中

```python
experiment_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_dir = experiment_time
os.makedirs(f"./logs/{experiment_dir}", exist_ok=True)
```

可以修改来定义保存log和模型的文件名。此外

```python
if __name__ == '__main__':
    # change the AI model by editing the init_file
    training_pipeline = TrainPipeline(init_model='best_policy.model')
    training_pipeline.run()
```

可以决定是否加载预训练模型。最后

```python
if (self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000):
	self.pure_mcts_playout_num += 1000
	self.best_win_ratio = self.best_win_ratio * 0.8
```

可以改变0.8的值调整每次是否保留模型的胜率阈值。其余超参的调整见报告。

#### 测试

```powershell
python ai_play.py
```

其中需要注意的是

```python
def run():
    n = 5
    width, height = 8, 8
    # change the AI model by editing the model_file
    model_file = 'best_policy.model'
```

通过更改`model_file`的值，可以选择想要使用的模型AI。另外

```python
mcts_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=2000)
```

通过调整两个`n_playout`的值可以更改生成每步的模拟速度，从而调整难度。建议第一个`n_playout`的值固定为400，否则在下棋时思考时间会过长。