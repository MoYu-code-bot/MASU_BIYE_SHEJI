package com.captain.service.impl;

import com.captain.entity.po.Project;
import com.captain.mapper.ProjectMapper;
import com.captain.service.ProjectService;
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
public class ProjectServiceImpl extends ServiceImpl<ProjectMapper, Project> implements ProjectService {

}
