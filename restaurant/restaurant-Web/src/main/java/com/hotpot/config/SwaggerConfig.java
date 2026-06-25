package com.hotpot.config;

import com.github.xiaoymin.knife4j.spring.annotations.EnableKnife4j;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Info;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import springfox.documentation.builders.ApiInfoBuilder;
import springfox.documentation.builders.PathSelectors;
import springfox.documentation.builders.RequestHandlerSelectors;
import springfox.documentation.oas.annotations.EnableOpenApi;
import springfox.documentation.service.ApiInfo;
import springfox.documentation.service.AuthorizationScope;
import springfox.documentation.service.Contact;
import springfox.documentation.service.HttpAuthenticationScheme;
import springfox.documentation.service.SecurityReference;
import springfox.documentation.service.SecurityScheme;
import springfox.documentation.spi.DocumentationType;
import springfox.documentation.spi.service.contexts.SecurityContext;
import springfox.documentation.spring.web.plugins.Docket;

import java.util.Collections;
import java.util.List;

@Configuration
@EnableOpenApi
@EnableKnife4j
public class SwaggerConfig {

    @Bean
    public OpenAPI customOpenAPI() {
        return new OpenAPI()
                .info(new Info()
                        .title("火锅到家 - 接口文档")
                        .version("1.0.0")
                        .description("火锅到家 RESTful API 文档"));
    }

    @Bean
    public Docket cApiDocket() {
        return new Docket(DocumentationType.OAS_30)
                .groupName("C端-顾客接口")
                .apiInfo(apiInfo())
                .select()
                .apis(RequestHandlerSelectors.basePackage("com.hotpot.controller.api"))
                .paths(PathSelectors.ant("/api/**"))
                .build()
                .securitySchemes(securitySchemes())
                .securityContexts(securityContexts());
    }

    @Bean
    public Docket bApiDocket() {
        return new Docket(DocumentationType.OAS_30)
                .groupName("B端-管理后台接口")
                .apiInfo(apiInfo())
                .select()
                .apis(RequestHandlerSelectors.basePackage("com.hotpot.controller.admin"))
                .paths(PathSelectors.ant("/admin/**"))
                .build()
                .securitySchemes(securitySchemes())
                .securityContexts(securityContexts());
    }

    private ApiInfo apiInfo() {
        return new ApiInfoBuilder()
                .title("火锅到家 - 接口文档")
                .description("火锅到家 RESTful API 文档")
                .version("1.0.0")
                .contact(new Contact("火锅到家开发团队", "", ""))
                .build();
    }

    private List<SecurityScheme> securitySchemes() {
        return Collections.singletonList(
                HttpAuthenticationScheme.JWT_BEARER_BUILDER
                        .name("Bearer")
                        .description("JWT 认证")
                        .build()
        );
    }

    private List<SecurityContext> securityContexts() {
        return Collections.singletonList(
                SecurityContext.builder()
                        .securityReferences(defaultAuth())
                        .build()
        );
    }

    private List<SecurityReference> defaultAuth() {
        AuthorizationScope authorizationScope = new AuthorizationScope("global", "全局访问");
        AuthorizationScope[] authorizationScopes = {authorizationScope};
        return Collections.singletonList(
                new SecurityReference("Bearer", authorizationScopes)
        );
    }
}
