package com.captain.service.impl;

import com.alibaba.fastjson.JSON;
import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.captain.entity.po.Permission;
import com.captain.entity.po.Role;
import com.captain.entity.po.RolePermission;
import com.captain.entity.po.UserRole;
import com.captain.entity.vo.RoleVo;
import com.captain.mapper.PermissionMapper;
import com.captain.mapper.RoleMapper;
import com.captain.service.PermissionService;
import com.captain.service.RolePermissionService;
import com.captain.service.RoleService;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.captain.service.UserRoleService;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;

/**
 * <p>
 *  服务实现类
 * </p>
 *
 * @author lianhong
 * @since 2020-08-22
 */
@Service
public class RoleServiceImpl extends ServiceImpl<RoleMapper, Role> implements RoleService {
   @Autowired
    PermissionService permissionService;
   @Autowired
   RolePermissionService rolePermissionService;
   @Autowired
    UserRoleService userRoleService;

    @Override
    public List<RoleVo> getListByUid(Long uid) {
        return baseMapper.getListByUid(uid);
    }

    @Override
    public Page getList(Page page) {
        List<RoleVo> list = baseMapper.getList(page, new QueryWrapper());
        return new Page().setRecords(list);
    }

    @Override
    public boolean saveRoleVo(RoleVo roleVo) {
        Role role=new Role();
        BeanUtils.copyProperties(roleVo,role);
        int insert = baseMapper.insert(role);
        if(insert==0) return false;
        //添加权限
        if(roleVo.getPermissions()==null||roleVo.getPermissions().size()==0) return true;
        List<RolePermission> insertPerms=new ArrayList<>();
        List<Permission> newPerms=roleVo.getPermissions();
        for(int i=0;i<newPerms.size();i++){
            Permission permission=newPerms.get(i);
            RolePermission rolePermission = new RolePermission();
            System.out.println(JSON.toJSONString(rolePermission));
            rolePermission.setPermissionId(permission.getId());
            rolePermission.setRoleId(role.getId());
            insertPerms.add(rolePermission);
        }
        if(insertPerms.size()>0){
            boolean b = rolePermissionService.saveBatch(insertPerms);
            if(!b) return false;
        }
        return true;
    }

    @Override
    public boolean updateRoleVo(RoleVo roleVo) {
        Role role=new Role();
        BeanUtils.copyProperties(roleVo,role);
        //更新角色
        int i = baseMapper.updateById(role);
        if(i==0) return false;
        if(roleVo.getPermissions()==null || roleVo.getPermissions().size()==0){
            return true;
        }
        //更新权限
        //获取旧的权限
        List<RolePermission> oldPerms = rolePermissionService.list(new QueryWrapper<RolePermission>()
        .eq("role_id",role.getId()));
        //获取新的权限
        List<Permission> newPerms=roleVo.getPermissions();
        List<RolePermission> insertPerms=new ArrayList<>();
        List<Long> deletePerms=new ArrayList<>();
        //找出需要新增的权限
        for(Permission item:newPerms){
            boolean needInsert=true;
            for(RolePermission old:oldPerms){
                if(old.getPermissionId().equals(item.getId())){
                    needInsert=false;
                    break;
                }
            }
            if(needInsert){
                RolePermission rolePermission = new RolePermission();
                rolePermission.setPermissionId(item.getId());
                rolePermission.setRoleId(role.getId());
                insertPerms.add(rolePermission);
            }
        }
        //新增
        if(insertPerms.size()>0){
            boolean b1 = rolePermissionService.saveBatch(insertPerms);
            if(!b1) return false;
        }
        //找出需要删除的
        for(RolePermission old:oldPerms){
            boolean needDelete=true;
            for(Permission item:newPerms){
                if(old.getPermissionId().equals(item.getId())){
                    needDelete=false;
                    break;
                }
            }
            if(needDelete) deletePerms.add(old.getId());
        }
        if(deletePerms.size()>0){
            //删除
            boolean b = rolePermissionService.removeByIds(deletePerms);
            return b;
        }
        return true;
    }

    @Override
    public boolean deleteRole(Long id) {
        //删除角色权限表的相关数据
        boolean rolePermRes = rolePermissionService.remove(new QueryWrapper<RolePermission>()
                .eq("role_id", id));
        //删除用户角色表的对应关系
        boolean userRoleRes = userRoleService.remove(new QueryWrapper<UserRole>()
                .eq("role_id", id));
        //删除角色表对应数据
        int roleRes = baseMapper.delete(new QueryWrapper<Role>()
                .eq("id", id));
        return true;
    }
}
