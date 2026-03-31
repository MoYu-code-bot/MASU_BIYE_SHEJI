package variable;

import java.util.Scanner;

public class VariableDemo04 {
    public static void main(String[] args) {
        /*
        练习 案例 17：牛牛数数（课堂练习，算法小题，学生自己独立完成）

        描述
        牛牛在酒桌上玩一个小游戏，第一个人从 1 开始数数，如果遇到数字中含有数字 4 或数字是 4 的倍数，则跳过这个数字报下一个，谁数错了就要罚酒一杯。
        牛牛为了作弊，它想将所有符合规则的数字预先生成出来。请你帮助牛牛列出 1 到 n 之间所有既不包含数字 4 又不是 4 的倍数的整数，按升序输出。

        输入描述
        在一行中输入一个正整数 n，满足 1 ≤ n ≤ 10⁵。

        输出描述
        按升序输出所有满足条件的整数，每个数字占一行。

        示例 1
        输入：9
        输出：
        1
        2
        3
        5
        6
        7
        9
        说明：在 1 到 9 中，数字 4 含有数字 4 且 4,8 为 4 的倍数，应跳过，剩余数字按升序输出。
        */
        Scanner sc = new Scanner(System.in);
        System.out.print("请输入正整数：");
        int n = sc.nextInt();

            for (int i = 1; i <= n; i++) {
                if (i % 4 == 0) {
                    continue;
                }
                boolean num = false;
                int temp = i;
                while (temp > 0){
                    if(temp % 10 == 4){
                        num = true;
                        break;
                    }
                    temp = temp / 10;

                }
                if(num){
                    continue;
                }
                System.out.println(i);
            }

    }
}
