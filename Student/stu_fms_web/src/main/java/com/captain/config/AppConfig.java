package com.captain.config;

import com.alibaba.fastjson.serializer.SerializerFeature;
import com.alibaba.fastjson.support.config.FastJsonConfig;
import com.alibaba.fastjson.support.spring.FastJsonHttpMessageConverter;
import com.captain.utils.FileUploadUtil;
import org.springframework.boot.autoconfigure.http.HttpMessageConverters;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.format.FormatterRegistry;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class AppConfig implements WebMvcConfigurer {

    @Override
    public void addFormatters(FormatterRegistry registry) {
        registry.addConverter((new DateConvertConfig()));
    }

    /**
     * 配置虚拟路径
     * @param registry
     */
    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        registry.addResourceHandler(FileUploadUtil.getVirtualRootPath()+"/**")
                .addResourceLocations("file:///"+FileUploadUtil.getRealRootPath());
    }


    /**
     * 配置对象传递到前端时的序列化规则
     * @return
     */
    @Bean
    public HttpMessageConverters fastJsonHttpMessageConverters() {
        FastJsonHttpMessageConverter fastConverter = new FastJsonHttpMessageConverter();
        //2.添加fastjson的配置信息，比如：是否要格式化返回的json数据
        FastJsonConfig fastJsonConfig = new FastJsonConfig();
        fastJsonConfig.setSerializerFeatures(SerializerFeature.BrowserCompatible);
        fastJsonConfig.setDateFormat("yyyy-MM-dd");
        //3.在convert中添加配置信息
        fastConverter.setFastJsonConfig(fastJsonConfig);
        return new HttpMessageConverters(fastConverter);
    }


}
