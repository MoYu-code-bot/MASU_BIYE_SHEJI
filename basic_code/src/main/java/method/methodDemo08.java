package method;

import java.util.Random;

public class methodDemo08 {
    public static void main(String[] args) {
        /*题目描述
        给定 n 个非负整数表示每个宽度为 1 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。
        输入示例
        height = [0,1,0,2,1,0,1,3,2,1,2,1]
        输出示例
        6
        题目解释
        下面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。
*/

        int[] arr = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};

        //从左到右遍历数组
        //定义从左往右看的数据
        int[] leftMax = new int[arr.length];
        //定义第三方变量temp,记录当前最高的柱子
        int temp = arr[0];
        for (int i = 0; i < arr.length; i++) {
            if (temp > arr[i]) {
                leftMax[i] = temp;
            } else {
                leftMax[i] = arr[i];
                temp = arr[i];
            }
        }

        //从右到左遍历数组
        int[] rightMax = new int[arr.length];
        temp = arr[arr.length - 1];
        for (int i = arr.length - 1; i >= 0; i--) {
            if (temp > arr[i]) {
                rightMax[i] = temp;
            } else {
                rightMax[i] = arr[i];
                temp = arr[i];
            }
        }
        //取交集
        int[] result = new int[arr.length];
        for (int i = 0; i < rightMax.length; i++) {
            if (leftMax[i] < rightMax[i]) {
                result[i] = leftMax[i];

            } else {
                result[i] = rightMax[i];

            }
        }
        int sum = 0;
        for (int i = 0; i < result.length; i++) {
            sum += result[i];
        }
        for (int i = 0; i < arr.length; i++) {
            sum -= arr[i];
        }
        System.out.println(sum);
    }
}
