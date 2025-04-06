import numpy as np
from test import generate_nearby_states


class State:
    def __init__(self, state, directionFlag=None, parent=None, f=0, g=0):
        # 当前棋盘的状态
        self.state = state
        # directionFlag是上一步空格移动的方向的反向（即不能再移回去）
        self.direction = ['up', 'down', 'right', 'left']
        if directionFlag:
            self.direction.remove(directionFlag)
        # 设置该状态的父状态
        self.parent = parent
        # 设置估价函数值f = g + h
        self.f = f
        # 设置初始状态到当前状态的实际代价
        self.g = g

    # 获取当前空格/0能够移动的方向
    def getDirection(self):
        return self.direction

    # 设置到该状态的代价
    def setF(self, f):
        self.f = f
        return

    # 打印结果
    def showInfo(self):
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                print(self.state[i, j], end='  ')
            print("\n")
        print('->')
        return

    # 获取空格/0点位置
    def getZeroPos(self):
        postion = np.where(self.state == 0)
        return postion

    # 曼哈顿距离f = g + h，g = 1（错误）
    def getFunctionValue(self):
        # 在拷贝上操作
        cur_node = self.state.copy()
        fin_node = self.answer.copy()

        # 当前状态到目标状态的总代价
        dist = 0
        # 获取一维的长度
        N = len(cur_node)

        # 遍历所有点
        for i in range(N):
            for j in range(N):
                if cur_node[i][j] != fin_node[i][j]:
                    # 不考虑0的曼哈顿距离
                    if cur_node[i][j] != 0:
                        # 找到目标状态实际位置坐标
                        index = np.argwhere(fin_node == cur_node[i][j])
                        x = index[0][0]  # 最终x距离
                        y = index[0][1]  # 最终y距离
                        dist += (abs(x - i) + abs(y - j))

        # return dist + g
        return dist + self.g

    # 第二种启发函数
    def getFunctionValue1(self):
        # 在拷贝上操作
        cur_node = self.state.copy()
        fin_node = self.answer.copy()

        # 当前状态到目标状态的总代价
        dist = 0
        # 获取一维的长度
        N = len(cur_node)

        # 遍历所有点
        for i in range(N):
            for j in range(N):
                if cur_node[i][j] != 0:
                    if cur_node[i][j] != fin_node[i][j]:
                        dist += 1

        # return dist + g
        return dist + self.g

    # 第三种启发函数
    def getFunctionValue2(self):
        # 在拷贝上操作
        cur_node = self.state.copy()
        fin_node = self.answer.copy()

        # 当前状态到目标状态的总代价
        dist = 0
        # 获取一维的长度
        N = len(cur_node)

        # 遍历所有点
        for i in range(N):
            for j in range(N):
                if cur_node[i][j] != fin_node[i][j]:
                    # 不考虑0的曼哈顿距离
                    if cur_node[i][j] != 0:
                        # 找到目标状态实际位置坐标
                        index = np.argwhere(fin_node == cur_node[i][j])
                        x = index[0][0]  # 最终x距离
                        y = index[0][1]  # 最终y距离
                        dist += pow(pow(x - i, 2) + pow(y - j, 2), 0.5)

        # return dist + g
        return dist + self.g

    # 第四种启发函数
    def getFunctionValue3(self):
        return self.g

    # 决定下一步往哪走/生成子状态
    def nextStep(self):
        # 若没有方向可以探索，则返回空
        if not self.direction:
            return []

        # 子状态（即open表）
        subStates = []
        boarder = len(self.state) - 1

        # 获取0点位置
        x, y = self.getZeroPos()

        # 向左
        if 'left' in self.direction and y > 0:
            # 对空格进行移动
            s = self.state.copy()
            tmp = s[x, y - 1]
            s[x, y - 1] = s[x, y]
            s[x, y] = tmp

            # 得到一个子状态
            news = State(s, directionFlag='right', parent=self, g=self.g + 1)
            if switch == 0:
                news.setF(news.getFunctionValue())
            elif switch == 1:
                news.setF(news.getFunctionValue1())
            elif switch == 2:
                news.setF(news.getFunctionValue2())
            elif switch == 3:
                news.setF(news.getFunctionValue3())
            subStates.append(news)

        # 向上
        if 'up' in self.direction and x > 0:
            s = self.state.copy()
            tmp = s[x - 1, y]
            s[x - 1, y] = s[x, y]
            s[x, y] = tmp

            news = State(s, directionFlag='down', parent=self, g=self.g + 1)
            if switch == 0:
                news.setF(news.getFunctionValue())
            elif switch == 1:
                news.setF(news.getFunctionValue1())
            elif switch == 2:
                news.setF(news.getFunctionValue2())
            elif switch == 3:
                news.setF(news.getFunctionValue3())
            subStates.append(news)

        # 向下
        if 'down' in self.direction and x < boarder:
            s = self.state.copy()
            tmp = s[x + 1, y]
            s[x + 1, y] = s[x, y]
            s[x, y] = tmp

            news = State(s, directionFlag='up', parent=self, g=self.g + 1)
            if switch == 0:
                news.setF(news.getFunctionValue())
            elif switch == 1:
                news.setF(news.getFunctionValue1())
            elif switch == 2:
                news.setF(news.getFunctionValue2())
            elif switch == 3:
                news.setF(news.getFunctionValue3())
            subStates.append(news)

        # 向右
        if self.direction.count('right') and y < boarder:
            s = self.state.copy()
            tmp = s[x, y + 1]
            s[x, y + 1] = s[x, y]
            s[x, y] = tmp

            news = State(s, directionFlag='left', parent=self, g=self.g + 1)
            if switch == 0:
                news.setF(news.getFunctionValue())
            elif switch == 1:
                news.setF(news.getFunctionValue1())
            elif switch == 2:
                news.setF(news.getFunctionValue2())
            elif switch == 3:
                news.setF(news.getFunctionValue3())
            subStates.append(news)

        # 返回F值最小的下一个点
        # subStates.sort(key=compareNum)
        # return subStates[0]
        # 返回所有子状态
        return subStates

    # A* 迭代
    def solve(self):
        # openList
        openTable = []
        # closeList
        closeTable = []
        openTable.append(self)

        cnt = 0

        while len(openTable) > 0:
            # 先对open表进行排序
            # openTable.sort(key=compareNum)
            openTable = sorted(openTable, key=lambda x: (x.f, x.f - x.g))

            # 打印open表和closed表
            print("第%d次搜索" % cnt)

            print("open表内容：")
            for i in range(3):
                for s in openTable:
                    for j in range(3):
                        print(s.state[i, j], end=" ")
                    print("", end=" ")
                print()

            print("closed表内容：")
            for i in range(3):
                for s in closeTable:
                    for j in range(3):
                        print(s.state[i, j], end=" ")
                    print("", end=" ")
                print()

            print()
            cnt = cnt + 1

            # 拿出open表中第一个状态
            n = openTable.pop(0)
            # 加入close表
            closeTable.append(n)

            # 最佳移动的路径
            path = []

            # 判断当前状态是否和最终结果相同
            if (n.state == n.answer).all():
                # 找到了最终结果后通过不断找父状态来重现路径
                while n:
                    path.append(n)
                    n = n.parent
                # 由于是逆向找的，需要逆序
                path.reverse()
                return path, cnt

            # 当与结果不同时将子状态放入open表
            subStates = n.nextStep()
            for sub in subStates:
                key = True
                # 判断子状态是否已经讨论过
                for x in closeTable:
                    if (sub.state == x.state).all():
                        key = False
                        break
                if key:
                    openTable.append(sub)

        # python中while居然可以和else连用
        else:
            return None, 0


def compareNum(state):
    return state.f


if __name__ == '__main__':
    #originState = State(np.array([[1, 5, 3], [2, 4, 6], [7, 0, 8]]))
    originState = State(np.array([[2, 8, 3], [1, 6, 4], [7, 0, 5]]))

    # State.answer = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    State.answer = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])

    # 定义一个开关用于选择使用哪种启发式函数
    switch = 0

    s1 = State(state=originState.state)
    path, _ = s1.solve()
    if path:
        for node in path:
            node.showInfo()
        print(State.answer)
        print("Total steps is %d" % (len(path) - 1))

    ############################################################

    # State.answer = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
    #
    # # 生成目标状态可能生成的所有状态
    # initialState = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
    # states = generate_nearby_states(initialState, 15, 100)
    #
    # result = [0, 0, 0, 0]
    #
    # for s in states:
    #     originState = State(np.array(s))
    #     s1 = State(state=originState.state)
    #     for j in range(4):
    #         switch = j
    #         s1 = State(state=originState.state)
    #         path, cnt = s1.solve()
    #         result[j] += cnt
    #
    # for r in range(len(result)):
    #     result[r] /= len(states)
    #
    # print(result)

