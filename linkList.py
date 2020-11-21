

'''
    python知识：
    1、类(object):
        不继承object对象，只拥有了__doc__ , __module__ 和 自己定义的name变量
        python 3 中已经默认就帮你加载了object
    
    2、 yield生成器
        def foo():
            print("starting...")
            while True:
                res = yield 4
                print("res:",res)
            例子：   f = foo()
                    next(f) 会返回4，此时中断于 res = yield 4 后；
                    再次执行 next(f), 会接着执行 print("res:",res)然后又到该句之前；
                    执行 f.send(10)， 执行完，返回4，且打印res:10

    3、 list
            list.append(obj)    在列表末尾添加新的对象
            list.count(obj)     统计某个元素在列表中出现的次数
            list.extend(seq)    在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
            list.index(obj)     从列表中找出某个值第一个匹配项的索引位置
            list.insert(index, obj)     将对象插入列表
            list.pop(obj=list[-1])      移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
            list.remove(obj)    移除列表中某个值的第一个匹配项
            list.reverse()      反向列表中元素
            list.sort([func])   对原列表进行排序 二维list,  list.sort(key = lambda x:[x[0],-[x1]] ,reverse=False) #对0升序、1降序排序 
'''


class Node(object):
    """单链表的结点"""

    def __init__(self, item):
        # item存放数据元素
        self.item = item
        # next是下一个节点的标识
        self.next = None


class SingleLinkList(object):
    """单链表"""

    def __init__(self):
        self._head = None

    def is_empty(self):
        """判断链表是否为空"""
        return self._head is None

    def length(self):
        """链表长度"""
        # 初始指针指向head
        cur = self._head
        count = 0
        # 指针指向None 表示到达尾部
        while cur is not None:
            count += 1
            # 指针下移
            cur = cur.next
        return count

    def items(self):
        """遍历链表"""
        # 获取head指针
        cur = self._head
        # 循环遍历
        while cur is not None:
            # 返回生成器
            yield cur.item
            # 指针下移
            cur = cur.next

    def add(self, item):
        """向链表头部添加元素"""
        node = Node(item)
        # 新结点指针指向原头部结点
        node.next = self._head
        # 头部结点指针修改为新结点
        self._head = node

    def append(self, item):
        """尾部添加元素"""
        node = Node(item)
        # 先判断是否为空链表
        if self.is_empty():
            # 空链表，_head 指向新结点
            self._head = node
        else:
            # 不是空链表，则找到尾部，将尾部next结点指向新结点
            cur = self._head
            while cur.next is not None:
                cur = cur.next
            cur.next = node

    def insert(self, index, item):
        """指定位置插入元素"""
        # 指定位置在第一个元素之前，在头部插入
        if index <= 0:
            self.add(item)
        # 指定位置超过尾部，在尾部插入
        elif index > (self.length() - 1):
            self.append(item)
        else:
            # 创建元素结点
            node = Node(item)
            cur = self._head
            # 循环到需要插入的位置
            for i in range(index - 1):
                cur = cur.next
            node.next = cur.next
            cur.next = node

    def remove(self, item):
        """删除节点"""
        cur = self._head
        pre = None
        while cur is not None:
            # 找到指定元素
            if cur.item == item:
                # 如果第一个就是删除的节点
                if not pre:
                    # 将头指针指向头节点的后一个节点
                    self._head = cur.next
                else:
                    # 将删除位置前一个节点的next指向删除位置的后一个节点
                    pre.next = cur.next
                return True
            else:
                # 继续按链表后移节点
                pre = cur
                cur = cur.next

    def find(self, item):
        """查找元素是否存在"""
        return item in self.items()

def fib(max):
    n, a, b = 0, 0, 1
    while n < max:
        print(b)
        a, b = b, a + b
        n = n + 1
    return 'done'

print(g)

if __name__ == "__main__":
    # 创建链表
    link_list = SingleLinkList()
    # 向链表尾部添加数据
    for i in range(5):
        link_list.append(i)
    # 向头部添加数据
    link_list.add(6)

    # 遍历链表数据
    for i in link_list.items():
        print(i, end='\t')
    # 链表数据插入数据
    link_list.insert(3, 9)
    print('\n', list(link_list.items()))
    # 删除链表数据
    link_list.remove(0)
    # 查找链表数据
    print(link_list.find(4))


    ##yield
    # def foo():
    #     print("starting...")
    #     while True:
    #         res = yield 4
    #         print("res:",res)
    # g = foo()
    # print(next(g))
    # print("*"*20)
    # print(next(g))
    # print("*"*20)
    # print(g.send(10))