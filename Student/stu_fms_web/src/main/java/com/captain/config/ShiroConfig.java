package com.captain.config;

import com.captain.realm.AuthorFilter;
import com.captain.realm.UserRealm;
import org.apache.shiro.mgt.SecurityManager;
import org.apache.shiro.spring.security.interceptor.AuthorizationAttributeSourceAdvisor;
import org.apache.shiro.spring.web.ShiroFilterFactoryBean;
import org.apache.shiro.web.mgt.DefaultWebSecurityManager;
import org.springframework.aop.framework.autoproxy.DefaultAdvisorAutoProxyCreator;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.servlet.Filter;
import java.util.HashMap;
import java.util.Map;

@Configuration
public class ShiroConfig {

    //开启shiro的aop
    @Bean
    @ConditionalOnMissingBean
    public DefaultAdvisorAutoProxyCreator defaultAdvisorAutoProxyCreator() {
        DefaultAdvisorAutoProxyCreator defaultAAP = new DefaultAdvisorAutoProxyCreator();
        defaultAAP.setProxyTargetClass(true);
        return defaultAAP;
    }

    //开启shiro的aop
    @Bean
    public AuthorizationAttributeSourceAdvisor authorizationAttributeSourceAdvisor(SecurityManager securityManager) {
        AuthorizationAttributeSourceAdvisor authorizationAttributeSourceAdvisor = new AuthorizationAttributeSourceAdvisor();
        authorizationAttributeSourceAdvisor.setSecurityManager(securityManager);
        return authorizationAttributeSourceAdvisor;
    }

    @Bean
    public ShiroFilterFactoryBean getShiroFilterFactory(DefaultWebSecurityManager securityManager){
        ShiroFilterFactoryBean filter=new ShiroFilterFactoryBean();
        filter.setSecurityManager(securityManager);
        /****************
         * 特别注意的就是配置拦截请求必须在设置登录页面上面
         * **************/
        //配置请求拦截
        //默认是anon：即是无需认证就允许通过
        Map<String,String> filterMap=new HashMap<>();
        filter.setFilterChainDefinitionMap(filterMap);

        //配置过滤器
        Map<String , Filter> map=new HashMap<>();
        //设置一个过滤器
        map.put("auth",new AuthorFilter());
        filter.setFilters(map);

        //未登录
        filter.setLoginUrl("/unLogin");
        //设置未授权
        filter.setUnauthorizedUrl("/unauthorized");

        return filter;
    }

    @Bean
    public DefaultWebSecurityManager securityManager(UserRealm userRealm){
        DefaultWebSecurityManager securityManager=new DefaultWebSecurityManager();
        securityManager.setRealm(userRealm);
        return securityManager;
    }


    @Bean
    public UserRealm userRealm(){
        return new UserRealm();
    }

}
