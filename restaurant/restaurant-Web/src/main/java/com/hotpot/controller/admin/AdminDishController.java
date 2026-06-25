package com.hotpot.controller.admin;

import com.hotpot.common.PageResult;
import com.hotpot.common.Result;
import com.hotpot.dto.PageQuery;
import com.hotpot.entity.Dish;
import com.hotpot.service.DishService;
import io.swagger.annotations.Api;
import io.swagger.annotations.ApiImplicitParam;
import io.swagger.annotations.ApiOperation;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@Api(tags = "B端-菜品管理")
@RestController
@RequestMapping("/admin/dishes")
@RequiredArgsConstructor
public class AdminDishController {

    private final DishService dishService;

    @ApiOperation("分页查询菜品")
    @GetMapping
    public Result<PageResult<Dish>> page(PageQuery query,
                                         @RequestParam(required = false) Long categoryId,
                                         @RequestParam(required = false) Long storeId,
                                         @RequestParam(required = false) Integer status) {
        return Result.success(dishService.pageQuery(query, categoryId, storeId, status));
    }

    @ApiOperation("新增菜品")
    @PostMapping
    public Result<?> add(@RequestBody Dish dish) {
        dishService.save(dish);
        return Result.success();
    }

    @ApiOperation("修改菜品")
    @ApiImplicitParam(name = "dishId", value = "菜品ID", required = true, dataType = "long", paramType = "query")
    @PutMapping("/update")
    public Result<?> update(@RequestParam Long dishId, @RequestBody Dish dish) {
        dish.setId(dishId);
        dishService.updateById(dish);
        return Result.success();
    }

    @ApiOperation("菜品上下架")
    @PutMapping("/updateStatus")
    public Result<?> updateStatus(@RequestParam Long dishId, @RequestParam Integer status) {
        Dish dish = new Dish();
        dish.setId(dishId);
        dish.setIsOnSale(status);
        dishService.updateById(dish);
        return Result.success();
    }

    @ApiOperation("删除菜品")
    @ApiImplicitParam(name = "dishId", value = "菜品ID", required = true, dataType = "long", paramType = "query")
    @DeleteMapping("/delete")
    public Result<?> delete(@RequestParam Long dishId) {
        dishService.removeById(dishId);
        return Result.success();
    }
}
