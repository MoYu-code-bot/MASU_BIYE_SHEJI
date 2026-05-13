package com.captain.config;

import com.baomidou.mybatisplus.core.handlers.MetaObjectHandler;
import com.baomidou.mybatisplus.core.incrementer.IdentifierGenerator;
import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.baomidou.mybatisplus.extension.plugins.PaginationInterceptor;
import org.apache.ibatis.reflection.MetaObject;
import org.mybatis.spring.annotation.MapperScan;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Date;

@Configuration
@MapperScan("com.captain.mapper")
public class MybatisPlusConfig {

    /**
     * 自动填充字段
     * 定义需要填充的字段
     * @return
     */
    @Bean
    public MetaObjectHandler metaObjectHandler(){
        return new MetaObjectHandler() {
            @Override
            public void insertFill(MetaObject metaObject) {
                this.strictInsertFill(metaObject,"id", Long.class, IdWorker.getId());
                this.strictInsertFill(metaObject,"created", Date.class,new Date());
                this.strictInsertFill(metaObject,"modified", Date.class,new Date());
                this.strictInsertFill(metaObject,"deleted", Integer.class,0);
            }

            @Override
            public void updateFill(MetaObject metaObject) {
                this.strictInsertFill(metaObject,"modified", Date.class,new Date());
            }

        };
    }

    /**
     * 配置分页拦截器
     * @return
     */
    @Bean
    public PaginationInterceptor paginationInterceptor() {
        return new PaginationInterceptor();
    }
}
