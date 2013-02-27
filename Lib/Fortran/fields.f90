module fields
use utils
implicit none

contains

subroutine grad_1d(a, g, dx)
    real(dp), intent(in) :: a(:), dx
    real(dp), intent(out) :: g(1, size(a, 1))
    integer :: x

    do x = 1, size(a, 1)
        g(1, x) = a(i_wrap(x + 1, size(a, 1))) - a(i_wrap(x - 1, size(a, 1)))
    end do
    g = g / (2.0_dp * dx)
end subroutine

subroutine grad_2d(a, g, dx)
    real(dp), intent(in) :: a(:, :), dx
    real(dp), intent(out) :: g(2, size(a, 1), size(a, 2))
    integer :: x, y

    do y = 1, size(a, 2)
        do x = 1, size(a, 1)
            g(1, x, y) = a(i_wrap(x + 1, size(a, 1)), y) - a(i_wrap(x - 1, size(a, 1)), y)
            g(2, x, y) = a(x, i_wrap(y + 1, size(a, 2))) - a(x, i_wrap(y - 1, size(a, 2)))
        end do
    end do
    g = g / (2.0_dp * dx)
end subroutine

subroutine grad_3d(a, g, dx)
    real(dp), intent(in) :: a(:, :, :), dx
    real(dp), intent(out) :: g(3, size(a, 1), size(a, 2), size(a, 3))
    integer :: x, y, z

    do z = 1, size(a, 3)
        do y = 1, size(a, 2)
            do x = 1, size(a, 1)
                g(1, x, y, z) = a(i_wrap(x + 1, size(a, 1)), y, z) - a(i_wrap(x - 1, size(a, 1)), y, z)
                g(2, x, y, z) = a(x, i_wrap(y + 1, size(a, 2)), z) - a(x, i_wrap(y - 1, size(a, 2)), z)
                g(3, x, y, z) = a(x, y, i_wrap(z + 1, size(a, 3))) - a(x, y, i_wrap(z - 1, size(a, 3)))
            end do
        end do
    end do
    g = g / (2.0_dp * dx)
end subroutine

subroutine grad_i_1d(a, r, l, gi)
    real(dp), intent(in) :: a(:), r(:, :), l
    real(dp), intent(out) :: gi(size(r, 1), size(r, 2))
    integer :: inds(size(r, 1), size(r, 2)), i, x
    real(dp) :: dx

    if (size(r, 1) /= 1) stop

    inds = r_to_inds(r, l, size(a, 1))
    do i = 1, size(r, 2)
        x = inds(1, i)
        gi(1, i) = a(i_wrap(x + 1, size(a, 1))) - a(i_wrap(x - 1, size(a, 1)))
    end do
    dx = l / size(a, 1)
    gi = gi / (2.0_dp * dx)
end subroutine

subroutine grad_i_2d(a, r, l, gi)
    real(dp), intent(in) :: a(:, :), r(:, :), l
    real(dp), intent(out) :: gi(size(r, 1), size(r, 2))
    integer :: inds(size(r, 1), size(r, 2)), i, x, y
    real(dp) :: dx

    if (size(r, 1) /= 2) stop

    inds = r_to_inds(r, l, size(a, 1))
    do i = 1, size(r, 2)
        x = inds(1, i)
        y = inds(2, i)
        gi(1, i) = a(i_wrap(x + 1, size(a, 1)), y) - a(i_wrap(x - 1, size(a, 1)), y)
        gi(2, i) = a(x, i_wrap(y + 1, size(a, 2))) - a(x, i_wrap(y - 1, size(a, 2)))
    end do
    dx = l / size(a, 1)
    gi = gi / (2.0_dp * dx)
end subroutine

subroutine grad_i_3d(a, r, l, gi)
    real(dp), intent(in) :: a(:, :, :), r(:, :), l
    real(dp), intent(out) :: gi(size(r, 1), size(r, 2))
    integer :: inds(size(r, 1), size(r, 2)), i, x, y, z
    real(dp) :: dx

    if (size(r, 1) /= 3) stop

    inds = r_to_inds(r, l, size(a, 1))
    do i = 1, size(inds, 2)
        x = inds(1, i)
        y = inds(2, i)
        z = inds(3, i)
        gi(1, i) = a(i_wrap(x + 1, size(a, 1)), y, z) - a(i_wrap(x - 1, size(a, 1)), y, z)
        gi(2, i) = a(x, i_wrap(y + 1, size(a, 2)), z) - a(x, i_wrap(y - 1, size(a, 2)), z)
        gi(3, i) = a(x, y, i_wrap(z + 1, size(a, 3))) - a(x, y, i_wrap(z - 1, size(a, 3)))
    end do
    dx = l / size(a, 1)
    gi = gi / (2.0_dp * dx)
end subroutine

subroutine div_1d(a, d, dx)
    real(dp), intent(in) :: a(:, :), dx
    real(dp), intent(out) :: d(size(a, 2))
    integer :: x

    do x = 1, size(a, 2)
        d(x) = a(1, i_wrap(x + 1, size(a, 2))) - a(1, i_wrap(x - 1, size(a, 2)))
    end do
    d = d / (2.0_dp * dx)
end subroutine

subroutine div_2d(a, d, dx)
    real(dp), intent(in) :: a(:, :, :), dx
    real(dp), intent(out) :: d(size(a, 2), size(a, 3))
    integer :: x, y

    do y = 1, size(a, 3)
        do x = 1, size(a, 2)
            d(x, y) = (a(1, i_wrap(x + 1, size(a, 2)), y) - a(1, i_wrap(x - 1, size(a, 2)), y)) &
                      + (a(2, x, i_wrap(y + 1, size(a, 3))) - a(2, x, i_wrap(y - 1, size(a, 3))))
        end do
    end do
    d = d / (2.0_dp * dx)
end subroutine

subroutine div_3d(a, d, dx)
    real(dp), intent(in) :: a(:, :, :, :), dx
    real(dp), intent(out) :: d(size(a, 2), size(a, 3), size(a, 4))
    integer :: x, y, z

    do z = 1, size(a, 4)
        do y = 1, size(a, 3)
            do x = 1, size(a, 2)
                d(x, y, z) = (a(1, i_wrap(x + 1, size(a, 2)), y, z) - a(1, i_wrap(x - 1, size(a, 2)), y, z)) &
                             + (a(2, x, i_wrap(y + 1, size(a, 3)), z) - a(2, x, i_wrap(y - 1, size(a, 3)), z)) &
                             + (a(3, x, y, i_wrap(z + 1, size(a, 4))) - a(3, x, y, i_wrap(z - 1, size(a, 4))))
            end do
        end do
    end do
    d = d / (2.0_dp * dx)
end subroutine

subroutine laplace_1d(a, l, dx)
    real(dp), intent(in) :: a(:), dx
    real(dp), intent(out) :: l(size(a, 1))
    integer :: x

    do x = 1, size(a, 1)
        l(x) = a(i_wrap(x + 1, size(a, 1))) + a(i_wrap(x - 1, size(a, 1)))
    end do
    l = (l - 2.0_dp * a) / (dx ** 2.0_dp)
end subroutine

subroutine laplace_2d(a, l, dx)
    real(dp), intent(in) :: a(:, :), dx
    real(dp), intent(out) :: l(size(a, 1), size(a, 2))
    integer :: x, y

    do y = 1, size(a, 2)
        do x = 1, size(a, 1)
            l(x, y) = a(i_wrap(x + 1, size(a, 1)), y) + a(i_wrap(x - 1, size(a, 1)), y) &
                      + a(x, i_wrap(y + 1, size(a, 2))) + a(x, i_wrap(y - 1, size(a, 2)))
        end do
    end do
    l = (l - 4.0_dp * a) / (dx ** 2.0_dp)
end subroutine

subroutine laplace_3d(a, l, dx)
    real(dp), intent(in) :: a(:, :, :), dx
    real(dp), intent(out) :: l(size(a, 1), size(a, 2), size(a, 3))
    integer :: x, y, z

    do z = 1, size(a, 3)
        do y = 1, size(a, 2)
            do x = 1, size(a, 1)
                l(x, y, z) = a(i_wrap(x + 1, size(a, 1)), y, z) + a(i_wrap(x - 1, size(a, 1)), y, z) &
                             + a(x, i_wrap(y + 1, size(a, 2)), z) + a(x, i_wrap(y - 1, size(a, 2)), z) &
                             + a(x, y, i_wrap(z + 1, size(a, 3))) + a(x, y, i_wrap(z - 1, size(a, 3)))
            end do
        end do
    end do
    l = (l - 6.0_dp * a) / (dx ** 2.0_dp)
end subroutine

subroutine density_1d(r, l, a)
    real(dp), intent(in) :: r(:, :), l
    integer, intent(inout) :: a(:)
    integer :: i, inds(size(r, 1), size(r, 2))

    if (size(r, 1) /= 1) stop

    inds = r_to_inds(r, l, size(a, 1))
    a = 0
    do i = 1, size(inds, 2)
        a(inds(1, i)) = a(inds(1, i)) + 1
    end do
end subroutine

subroutine density_2d(r, l, a)
    real(dp), intent(in) :: r(:, :), l
    integer, intent(inout) :: a(:, :)
    integer :: i, inds(size(r, 1), size(r, 2))

    if (size(r, 1) /= 2) stop

    inds = r_to_inds(r, l, size(a, 1))
    a = 0
    do i = 1, size(inds, 2)
        a(inds(1, i), inds(2, i)) = a(inds(1, i), inds(2, i)) + 1
    end do
end subroutine

subroutine density_3d(r, l, a)
    real(dp), intent(in) :: r(:, :), l
    integer, intent(inout) :: a(:, :, :)
    integer :: i, inds(size(r, 1), size(r, 2))

    if (size(r, 1) /= 3) stop

    inds = r_to_inds(r, l, size(a, 1))
    a = 0
    do i = 1, size(inds, 2)
        a(inds(1, i), inds(2, i), inds(3, i)) = a(inds(1, i), inds(2, i), inds(3, i)) + 1
    end do
end subroutine

end module
