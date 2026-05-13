package com.captain.config;

import com.alibaba.druid.pool.DruidDataSource;
import com.alibaba.druid.support.http.StatViewServlet;
import com.alibaba.druid.support.http.WebStatFilter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.web.servlet.FilterRegistrationBean;
import org.springframework.boot.web.servlet.ServletRegistrationBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.sql.DataSource;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@Configuration
public class DruidConfig {
    /**
     * 创建一个druid的数据源，相关配置来源于
     * 配置类里面的spring.datasource
     * @return
     */
    @ConfigurationProperties(prefix = "spring.datasource")
    @Bean
    public DataSource createDruidDataSource(){
        return new DruidDataSource();
    }

    /****************配置druid后台监控*****************/
    @Bean
    public ServletRegistrationBean statViewServlet(){
        ServletRegistrationBean registrationBean=new ServletRegistrationBean(new StatViewServlet(),"/druid/*");
        //配置后台监控的用户信息
        Map<String,String> druidParams=new HashMap<>();
        druidParams.put("loginUsername","admin");
        druidParams.put("loginPassword","123456");
        druidParams.put("allow","");
        druidParams.put("deny","47.94.227.156");
        registrationBean.setInitParameters(druidParams);
        return registrationBean;
    }

    //配置一个过滤器
    @Bean
    public FilterRegistrationBean webStatFilter(){
        FilterRegistrationBean  bean=new FilterRegistrationBean();
        bean.setFilter(new WebStatFilter());
        Map<String,String> map=new HashMap<>();
        map.put("exclusions","*.js,*.css,/druid/*");
        bean.setInitParameters(map);
        bean.setUrlPatterns(Arrays.asList("/*"));
        return bean;
    }
}
