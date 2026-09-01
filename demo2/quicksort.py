def quicksort(arr):
    """快速排序（递归实现，返回新列表）"""
    if len(arr) <= 1:
        return arr
    pivot = arr[0]
    less = [x for x in arr[1:] if x <= pivot]
    greater = [x for x in arr[1:] if x > pivot]
    return quicksort(less) + [pivot] + quicksort(greater)


def quicksort_inplace(arr, low=0, high=None):
    """快速排序（原地实现，内存更省）"""
    if high is None:
        high = len(arr) - 1

    def partition(arr, low, high):
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    if low < high:
        pi = partition(arr, low, high)
        quicksort_inplace(arr, low, pi - 1)
        quicksort_inplace(arr, pi + 1, high)
    return arr


if __name__ == "__main__":
    # 测试用例
    test = [3, 6, 8, 10, 1, 2, 1]
    print("原始:", test)

    # 方式一：返回新列表
    print("排序(新列表):", quicksort(test))

    # 方式二：原地排序
    test2 = [3, 6, 8, 10, 1, 2, 1]
    quicksort_inplace(test2)
    print("排序(原地):", test2)
