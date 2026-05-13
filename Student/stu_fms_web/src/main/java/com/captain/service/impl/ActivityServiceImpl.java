package com.captain.service.impl;

import com.captain.entity.po.Activity;
import com.captain.mapper.ActivityMapper;
import com.captain.service.ActivityService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import org.springframework.stereotype.Service;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author lianhong
 * @since 2020-08-24
 */
@Service
public class ActivityServiceImpl extends ServiceImpl<ActivityMapper, Activity> implements ActivityService {

}
