package com.captain;

import com.baomidou.mybatisplus.annotation.DbType;
import com.baomidou.mybatisplus.annotation.IdType;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.toolkit.StringPool;
import com.baomidou.mybatisplus.generator.AutoGenerator;
import com.baomidou.mybatisplus.generator.InjectionConfig;
import com.baomidou.mybatisplus.generator.config.*;
import com.baomidou.mybatisplus.generator.config.po.TableInfo;
import com.baomidou.mybatisplus.generator.config.rules.DateType;
import com.baomidou.mybatisplus.generator.config.rules.NamingStrategy;
import com.captain.entity.po.BaseEntity;
import com.captain.entity.po.User;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.env.Environment;
import org.springframework.util.StringUtils;

import javax.sql.DataSource;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class CodeGenerator {

    /**
     * 代码生成
     */
    public static void main(String args[]) {
        //创建一个代码自动生成器
        AutoGenerator autoGenerator = new AutoGenerator();
        Scanner scanner=new Scanner(System.in);
        //全局配置
        GlobalConfig globalConfig = new GlobalConfig();
        //获取作者的名字
        System.out.println("作者名字使用(lianhong),y/n?");
        String res=scanner.nextLine();
        globalConfig.setAuthor("lianhong");
        if(!res.equals("y")){
            System.out.println("请输入作者名字：");
            String authorName=scanner.nextLine();
            globalConfig.setAuthor(authorName);
        }
        String objectPath=System.getProperty("user.dir");
        globalConfig.setOutputDir(objectPath+"/src/main/java");
        globalConfig.setOpen(false);
        //设置不覆盖生成好了的,避免把代码冲掉
        globalConfig.setFileOverride(false);
        globalConfig.setServiceName("%sService");//去掉service层的I前缀
        globalConfig.setIdType(IdType.ASSIGN_ID);//使用雪花算法
        globalConfig.setDateType(DateType.ONLY_DATE);
        autoGenerator.setGlobalConfig(globalConfig);

        //配置数据源,直接使用yml配置的
        DataSourceConfig dataSourceConfig = new DataSourceConfig();
        dataSourceConfig.setUsername("root");
        dataSourceConfig.setPassword("5730lh520@");
        dataSourceConfig.setUrl("jdbc:mysql://47.94.227.156:3307/stu_fms?useUnicode=true&characterEncoding=utf-8&serverTimezone=GMT%2B8");
        dataSourceConfig.setDriverName("com.mysql.cj.jdbc.Driver");
        dataSourceConfig.setDbType(DbType.MYSQL);
        autoGenerator.setDataSource(dataSourceConfig);

        //配置生成的文件的位置
        PackageConfig packageConfig = new PackageConfig();
        packageConfig.setParent("com.captain");
        packageConfig.setEntity("entity.po");
        autoGenerator.setPackageInfo(packageConfig);

        //自定义xml生成位置
        InjectionConfig cfg = new InjectionConfig() {
            @Override
            public void initMap() {
                // to do nothing
            }
        };
        List<FileOutConfig> list=new ArrayList<>();
        list.add(new FileOutConfig("/templates/mapper.xml.vm") {
            @Override
            public String outputFile(TableInfo tableInfo) {
                return System.getProperty("user.dir") +
                        "/src/main/resources/static/mapper/" +
                        tableInfo.getEntityName() + "Mapper" + StringPool.DOT_XML;
            }
        });
        cfg.setFileOutConfigList(list);
        autoGenerator.setCfg(cfg);

        //生成的模板路径位置
        TemplateConfig templateConfig=new TemplateConfig();
        //配置不生成controller

        templateConfig.setController(null);
        templateConfig.setXml(null);
        autoGenerator.setTemplate(templateConfig);

        // 策略配置
        StrategyConfig strategy = new StrategyConfig();
        strategy.setNaming(NamingStrategy.underline_to_camel);
        strategy.setColumnNaming(NamingStrategy.underline_to_camel);
        strategy.setEntityLombokModel(true);
        strategy.setSuperEntityClass(BaseEntity.class);
//        strategy.setRestControllerStyle(true);

        //读取一行table名字
        System.out.println("请输入需要生成的表名(中间用英文逗号分隔):");
        String tablesStr=scanner.nextLine();
        if(!StringUtils.isEmpty(tablesStr)){
            String[] tables=tablesStr.split(",");
            strategy.setInclude(tables);
            autoGenerator.setStrategy(strategy);
            //执行这个代码自动生成器
            autoGenerator.execute();
        }
        System.out.println("生成结束!");
    }
}
