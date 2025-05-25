import * as React from "react";
import { Navbar as BootstrapNavbar, Container, Nav } from "react-bootstrap";
import { TbWavesElectricity } from "react-icons/tb";
import { Link } from "react-router";

const Navbar: React.FunctionComponent = () => {
  return (
    <BootstrapNavbar
      id="navbar"
      expand="lg"
      variant="light"
      bg="light"
      sticky="top"
    >
      <Container>
        <Nav.Link as={Link} to="/">
          <BootstrapNavbar.Brand>
            <img
              src="/logo.png"
              width="30"
              height="30"
              className="d-inline-block align-top"
              alt="Logo"
            />
          </BootstrapNavbar.Brand>
        </Nav.Link>
        <BootstrapNavbar.Toggle>
          <span className="navbar-toggler-icon" />
        </BootstrapNavbar.Toggle>

        <BootstrapNavbar.Collapse>
          <Nav className="d-flex justify-content-between flex-row w-100">
            <div className="d-flex flex-wrap">
              <Nav.Link as={Link} to="/">
                <TbWavesElectricity /> Главная
              </Nav.Link>
            </div>
          </Nav>
        </BootstrapNavbar.Collapse>
      </Container>
    </BootstrapNavbar>
  );
};

export default Navbar;
