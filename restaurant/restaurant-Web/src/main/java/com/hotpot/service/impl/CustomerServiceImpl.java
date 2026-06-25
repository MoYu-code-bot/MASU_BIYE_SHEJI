package com.hotpot.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.hotpot.common.BusinessException;
import com.hotpot.entity.Customer;
import com.hotpot.mapper.CustomerMapper;
import com.hotpot.service.CustomerService;
import com.hotpot.util.JwtUtil;
import lombok.RequiredArgsConstructor;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class CustomerServiceImpl extends ServiceImpl<CustomerMapper, Customer> implements CustomerService {

    private final JwtUtil jwtUtil;
    private final BCryptPasswordEncoder passwordEncoder = new BCryptPasswordEncoder();

    @Override
    public String login(String phone, String password) {
        Customer customer = getByPhone(phone);
        if (customer == null) {
            throw new BusinessException("手机号未注册");
        }
        if (customer.getStatus() == 0) {
            throw new BusinessException("账号已被禁用");
        }
        if (!passwordEncoder.matches(password, customer.getPassword())) {
            throw new BusinessException("密码错误");
        }
        return jwtUtil.generateMemberToken(customer.getId(), customer.getPhone());
    }

    @Override
    public String register(String phone, String password, String nickname) {
        Customer existing = getByPhone(phone);
        if (existing != null) {
            throw new BusinessException("手机号已注册");
        }
        Customer customer = new Customer();
        customer.setPhone(phone);
        customer.setPassword(passwordEncoder.encode(password));
        customer.setNickname(nickname);
        customer.setGender(0);
        customer.setStatus(1);
        save(customer);
        return jwtUtil.generateMemberToken(customer.getId(), customer.getPhone());
    }

    @Override
    public Customer getByPhone(String phone) {
        return getOne(new LambdaQueryWrapper<Customer>()
                .eq(Customer::getPhone, phone));
    }
}
