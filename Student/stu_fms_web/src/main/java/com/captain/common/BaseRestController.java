package com.captain.common;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.captain.entity.po.Audit;
import com.captain.entity.po.BaseEntity;
import com.captain.service.AuditService;
import lombok.Data;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.util.Assert;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

import java.lang.reflect.Field;
import java.util.List;

/**
 * 顶层的controller
 *
 *
 * @author lianhong
 * @date 2020-8-24
 */
public class BaseRestController<T extends IService, E extends BaseEntity> {
    @Autowired
    protected T baseService;
    @Autowired
    protected AuditService auditService;

    /**
     * 获取搜索用的QueryWrapper(查询条件，需要自己去重载,不然就只有搜索全部)
     * @param searchValue 搜索的值
     * @param e 需要精确搜索的参数
     * @return
     */
    protected QueryWrapper<E> getSearchQueryWrapper(String searchValue,E e){
        return new QueryWrapper();
    }

    /**
     * 模糊搜索
     * @param page
     * @param size
     * @param searchValue 需要模糊搜索的值
     * @param e 需要精确搜索的参数
     * @return
     */
    @GetMapping("/search")
    public Result fuzzySearch(Integer page, Integer size,String searchValue,E e){
        Assert.notNull(page, "页数不能为空");
        Assert.notNull(size, "页大小不能为空");
        IPage pageInfo = baseService.page(new Page(page, size), getSearchQueryWrapper(searchValue,e));
        return Result.success(pageInfo);
    }

    /**
     * 分页获取
     *
     * @return
     */

    @PostMapping("/insert")
    public Result insertMapping(@RequestBody E e) throws NoSuchFieldException {
        Result result = insert(e);
        if(result.getStatus()==0){
            //insert之后就有id
            Result result1 = insertIntoAudit(e.getId());
            result.setStatus(result1.getStatus());
        }
        return result;
    }
    /**
     * 分页获取
     *
     * @return
     */
    @GetMapping("/list")
    public Result list(Integer page, Integer size) {
        Assert.notNull(page, "页数不能为空");
        Assert.notNull(size, "页大小不能为空");
        IPage pageInfo = baseService.page(new Page(page, size), null);
        return Result.success(pageInfo);
    }

    protected Result insert(E e){
        boolean res = baseService.save(e);
        if (!res) return Result.fail();
        return Result.success();
    }

    protected Result insertIntoAudit(Long id) throws NoSuchFieldException {
        Assert.notNull(id,"提交审核的档案ID不能为空");
        //重新从数据库去这个数据
        E e = (E) baseService.getById(id);
        //判断有没有档案类型这个字段，有那么就需要提交到审核部分
        Class<? extends BaseEntity> aClass = e.getClass();
        try{
            Field field = aClass.getDeclaredField("archivesTypeCode");
            if(field!=null){
                //提交到审核中心去审核
                Audit audit = new Audit();
                audit.setArchiveId(e.getId());
                field.setAccessible(true);
                audit.setArchiveTypeCode((Long) field.get(e));
                try{
                    Field stuId = aClass.getDeclaredField("stuId");
                    stuId.setAccessible(true);
                    audit.setStuId(String.valueOf(stuId.get(e)));
                }catch (Exception exception){
                    exception.printStackTrace();
                }
                audit.setIsAudited(0);
                boolean save = auditService.save(audit);
                if(!save) Result.fail();
            }
            return Result.success();
        }catch (Exception exception){
            exception.printStackTrace();
            return Result.success();
        }
    }

    /**
     * 新增
     *
     * @param e
     * @return
     */


    protected Result delete(E e){
        Assert.notNull(e.getId(), "id不能为空");
        boolean res = baseService.removeById(e.getId());
        if (res) {
            return Result.success();
        } else {
            return Result.fail();
        }
    }

    /**
     * 删除
     *
     * @param e
     * @return
     */
    @PostMapping("/delete")
    public Result deleteMapping(@RequestBody E e) {
        return delete(e);
    }

    protected Result deleteBatch(List<Long> ids){
        Assert.notNull(ids, "参数不能为空");
        boolean res = baseService.removeByIds(ids);
        if (res) {
            return Result.success();
        } else {
            return Result.fail();
        }
    }

    /**
     * 批量删除
     *
     * @param ids
     * @return
     */
    @PostMapping("/deleteBatch")
    public Result deleteBatchMapping(@RequestBody IdList ids) {
        return deleteBatch(ids.getIds());
    }

    protected Result update(E e){
        Assert.notNull(e.getId(), "id不能为空");
        boolean res = baseService.updateById(e);
        if (res) {
            return Result.success();
        } else {
            return Result.fail();
        }
    }

    /**
     * 更新
     *
     * @param e
     * @return
     */
    @PostMapping("/update")
    public Result updateMapping(@RequestBody E e) {
        return update(e);
    }

}
/**
 * ids的一个封装类
 * 解决application/json类型的数据传递
 */
@Data
class IdList {
    private List<Long> ids;
}

