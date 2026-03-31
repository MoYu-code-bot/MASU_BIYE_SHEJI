package variable;

public class VariableDemo08 {
    public static void main(String[] args) {
        //打印平行四边形
/*        for (int i = 0; i < 3; i++) {
            for (int j = i; j <= 2; j++) {
                System.out.print(" ");
            }
            for (int n = 0; n < 6; n++) {
                System.out.print("*");
            }
            System.out.println();
        }*/


        //打印菱形，中间一行为7个*（1,3,5,7,5,3,1）
        /*
                   *
                  ***
                 *****
                *******
                 *****
                  ***
                   *
        */
        /*
        for (int i = 1; i <= 4; i++) {
            for (int j = i; j < 4; j++) {
                System.out.print(" ");
            }
            for (int j = 1; j <= i; j++) {
                System.out.print("*");
            }
            for (int j = 0; j <= i - 2; j++) {
                System.out.print("*");
            }
            for (int j = i; j < 4; j++) {
                System.out.print(" ");

            }
            System.out.println();
        }
        for (int i = 1; i <= 3; i++) {
            for (int j = 1; j <= i; j++) {
                System.out.print(" ");
            }
            for (int j = i; j <= 3; j++) {
                System.out.print("*");
            }
            for (int j = i; j <= 2; j++) {
                System.out.print("*");
            }
            System.out.println();
        }
        */

        //打印空心菱形
        /*
                   *
                  * *
                 *   *
                  * *
                   *

        */
        // 上半部分（包括中间行）- 共 3 行
        for (int i = 1; i <= 3; i++) {
            // 打印前导空格
            for (int j = i; j < 3; j++) {
                System.out.print(" ");
            }
            // 打印第一个星号
            System.out.print("*");

            // 如果不是第一行，打印中间的空格和第二个星号
            if (i > 1) {
                // 打印中间空格：第 2 行 1 个空格，第 3 行 3 个空格
                for (int j = 1; j <= 2 * i - 3; j++) {
                    System.out.print(" ");
                }
                // 打印第二个星号
                System.out.print("*");
            }

            System.out.println();
        }

        // 下半部分 - 共 2 行
        for (int i = 1; i <= 2; i++) {
            // 打印前导空格
            for (int j = 1; j <= i; j++) {
                System.out.print(" ");
            }
            // 打印第一个星号
            System.out.print("*");

            // 如果不是最后一行，打印中间的空格和第二个星号
            if (i < 2) {
                // 打印中间空格：第 1 行 1 个空格
                for (int j = 1; j <= 2 * (2 - i) - 1; j++) {
                    System.out.print(" ");
                }
                // 打印第二个星号
                System.out.print("*");
            }

            System.out.println();
        }






    }
}
