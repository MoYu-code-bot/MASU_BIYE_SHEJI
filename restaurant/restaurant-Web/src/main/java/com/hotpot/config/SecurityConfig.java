package com.hotpot.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.security.config.annotation.method.configuration.EnableGlobalMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.UsernamePasswordAuthenticationFilter;
import org.springframework.web.cors.CorsConfiguration;
import org.springframework.web.cors.CorsConfigurationSource;
import org.springframework.web.cors.UrlBasedCorsConfigurationSource;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

@Configuration
@EnableWebSecurity
@EnableGlobalMethodSecurity(prePostEnabled = true)
@RequiredArgsConstructor
public class SecurityConfig {

    private final JwtTokenFilter jwtTokenFilter;

    @Bean
    public SecurityFilterChain filterChain(HttpSecurity http) throws Exception {
        http
                .cors().and()
                .csrf().disable()
                .sessionManagement().sessionCreationPolicy(SessionCreationPolicy.STATELESS)
                .and()
                .authorizeRequests()
                        // C端公开接口（浏览类无需登录）
                        .antMatchers("/api/stores/**").permitAll()
                        .antMatchers("/api/categories/**").permitAll()
                        .antMatchers("/api/banners/**").permitAll()
                        .antMatchers("/api/announcements/**").permitAll()
                        // C端需要登录的接口
                        .antMatchers("/api/auth/login", "/api/auth/register").permitAll()
                        .antMatchers("/api/auth/**").authenticated()
                        .antMatchers(HttpMethod.POST, "/api/reservations/**").authenticated()
                        .antMatchers(HttpMethod.GET, "/api/reservations/**").authenticated()
                        .antMatchers(HttpMethod.DELETE, "/api/reservations/**").authenticated()
                        .antMatchers("/api/reservations/**").authenticated()
                        .antMatchers(HttpMethod.POST, "/api/reviews/**").authenticated()
                        .antMatchers(HttpMethod.GET, "/api/reviews/**").permitAll()
                        // B端登录接口
                        .antMatchers("/admin/auth/login").permitAll()
                        // B端角色权限控制（按URL前缀）
                        .antMatchers("/admin/users/**").hasAnyRole("ADMIN", "MANAGER") // Admin+店长管理用户（Controller层细粒度控制：店长不可操作ADMIN）
                        .antMatchers("/admin/stores/**").hasAnyRole("ADMIN", "MANAGER") // Admin+店长管理门店
                        .antMatchers("/admin/reservations/**").hasAnyRole("ADMIN", "MANAGER", "STAFF") // 所有角色查看预订
                        .antMatchers("/admin/reviews/**").hasAnyRole("ADMIN", "MANAGER", "STAFF")      // 所有角色查看评价
                        .antMatchers("/admin/**").authenticated()                  // 其余B端需要认证
                        // Swagger / 静态资源
                        .antMatchers("/doc.html").permitAll()
                        .antMatchers("/swagger-resources/**").permitAll()
                        .antMatchers("/webjars/**").permitAll()
                        .antMatchers("/v3/**").permitAll()
                        .antMatchers("/static/**").permitAll()
                        .antMatchers("/favicon.ico").permitAll()
                        // 其余所有请求需要认证
                        .anyRequest().authenticated()
                .and()
                // 未认证时返回 JSON 401 而不是重定向或 403
                .exceptionHandling()
                        .authenticationEntryPoint(this::commenceUnauthorized)
                        .accessDeniedHandler(this::handleAccessDenied)
                .and()
                .addFilterBefore(jwtTokenFilter, UsernamePasswordAuthenticationFilter.class);

        return http.build();
    }

    /**
     * 处理未认证请求 → 返回 401 JSON
     */
    private void commenceUnauthorized(HttpServletRequest request,
                                       HttpServletResponse response,
                                       AuthenticationException authException) throws IOException {
        response.setStatus(HttpStatus.UNAUTHORIZED.value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setCharacterEncoding("UTF-8");

        Map<String, Object> body = new HashMap<>();
        body.put("code", 401);
        body.put("message", "未登录或登录已过期，请重新登录");
        body.put("data", null);

        new ObjectMapper().writeValue(response.getWriter(), body);
    }

    /**
     * 处理无权限访问 → 返回 403 JSON
     */
    private void handleAccessDenied(HttpServletRequest request,
                                      HttpServletResponse response,
                                      org.springframework.security.access.AccessDeniedException accessDeniedException) throws IOException {
        response.setStatus(HttpStatus.FORBIDDEN.value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setCharacterEncoding("UTF-8");

        Map<String, Object> body = new HashMap<>();
        body.put("code", 403);
        body.put("message", "没有访问权限");
        body.put("data", null);

        new ObjectMapper().writeValue(response.getWriter(), body);
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public CorsConfigurationSource corsConfigurationSource() {
        CorsConfiguration configuration = new CorsConfiguration();
        configuration.setAllowedOrigins(Arrays.asList("*"));
        configuration.setAllowedMethods(Arrays.asList("GET", "POST", "PUT", "DELETE", "OPTIONS"));
        configuration.setAllowedHeaders(Arrays.asList("*"));
        UrlBasedCorsConfigurationSource source = new UrlBasedCorsConfigurationSource();
        source.registerCorsConfiguration("/**", configuration);
        return source;
    }
}
