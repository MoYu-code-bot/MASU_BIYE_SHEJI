package ifDemo;

import java.util.Scanner;

public class ifDemo01 {
    public static void main(String[] args) {
        //满血200，受到伤害相应减少，最低血量为1
        int hp = 200;
        Scanner sc = new Scanner(System.in);
        System.out.println("请输入伤害：");
        int hurt = sc.nextInt();
        hp -= hurt;
        if (hp <= 0) {
            hp = 1;
        }
        System.out.println("剩余血量：" + hp);

        //受到伤害使用技能回血，最大血量不能超过200
        System.out.println("请输入技能回血：");
        int skill = sc.nextInt();
        hp += skill;
        if (hp > 200) {
            hp = 200;
        }
        System.out.println("当前血量：" + hp);

        if (hp == 200){
            System.out.println("当前血量已达到最大值！");
        }

    }

}
