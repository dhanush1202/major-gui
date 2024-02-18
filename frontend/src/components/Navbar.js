import { Navbar } from "keep-react";
import { useEffect, useState } from "react";
import { NavLink, useLocation } from "react-router-dom";

export const NavbarComponent = () => {
  const { pathname } = useLocation();
  const [path, setpath] = useState("/");
  useEffect(() => {
    setpath(pathname)
    // console.log(pathname)
  }, [pathname])
  return (
    <div className="w-[100vw] shadow-md">
      <Navbar fluid={true}>
        <Navbar.Container className="flex items-center justify-between lg:justify-center lg:h-[10vh]   ">
          <Navbar.Container className="flex items-center">
            <Navbar.Container
              tag="ul"
              className="lg:flex hidden items-center justify-between gap-8"
            >
              <NavLink
                to="/"
                className={`text-lg font-inter hover:border-b-2 duration-100 border-black ${path === "/" ? "font-semibold border-b-2" : ""}`}
              >
                Home
              </NavLink>
              <NavLink
                to="/input"
                className={`text-lg font-inter hover:border-b-2 duration-100 border-black ${path === "/input" ? "font-semibold border-b-2" : ""}`}
              >
                Input
              </NavLink>
            </Navbar.Container>
            <Navbar.Collapse collapseType="sidebar">
              <Navbar.Container tag="ul" className="flex flex-col gap-5">
                <NavLink
                  to="/"
                  className={`text-lg font-inter hover:border-b-2 duration-100 border-black w-fit  ${path === "/" ? "font-semibold border-b-2" : ""}`}  
                >
                  Home
                </NavLink>
                <NavLink
                  to="/input"
                  className={`text-lg font-inter hover:border-b-2 duration-100 border-black w-fit ${path === "/input" ? "font-semibold border-b-2" : ""}`}    
                >
                  Input
                </NavLink>
              </Navbar.Container>
            </Navbar.Collapse>
          </Navbar.Container>

          <Navbar.Container className="flex gap-2">
            <Navbar.Toggle />
          </Navbar.Container>
        </Navbar.Container>
      </Navbar>
    </div>
  );
};
