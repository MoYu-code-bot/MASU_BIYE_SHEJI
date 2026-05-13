package com.captain.realm;

import com.captain.common.Result;
import org.apache.shiro.web.filter.authc.UserFilter;

import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServletRequest;
import java.io.IOException;

/**
 * 当用户没有被认证的时候，shiro会调用这个方法
 */
public class AuthorFilter extends UserFilter {
    @Override
    protected void redirectToLogin(ServletRequest request, ServletResponse response) throws IOException {
        HttpServletRequest req=(HttpServletRequest)request;
        //获取请求头
        String head=req.getHeader("X-Requested-With");
        if(head!=null&&"XMLHttpRequest".equals(head)){
            //ajax就返回请求弹框
            response.setContentType("application/json;charset=UTF-8");
            response.getWriter().print(Result.fail("请先登录!"));
        }else{
            //如果是web请求（直接通过url访问）就返回到登录页面
            super.redirectToLogin(request, response);
        }

    }

}
